"""
RFEN2_RN_6.py - Neurona de Refuerzo con Pesos Cuánticos Híbridos
================================================================

Esta neurona implementa Pesos Cuánticos Híbridos que combinan computación
cuántica con redes neuronales clásicas, utilizando técnicas avanzadas de 2025
para optimización de pesos con superposición cuántica.

Características Avanzadas 2025:
- Pesos cuánticos con superposición de estados
- Computación cuántica híbrida clásica-cuántica
- Entrelazamiento cuántico entre pesos
- Interferencia cuántica para optimización
- Medición cuántica adaptativa
- Decoherencia controlada
- Optimización cuántica de pesos
- Algoritmos cuánticos de aprendizaje

Autor: LucIA Development Team
Versión: 2.0.0
Fecha: 2025
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
import logging
from collections import deque
import time
from . import NeuronaRefuerzoAvanzadaBase, inicializar_pesos_cuanticos, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_6')


class NeuronaRefuerzoCuantica(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con Pesos Cuánticos Híbridos.

    Esta neurona implementa pesos cuánticos que utilizan superposición
    cuántica y entrelazamiento para optimización avanzada de pesos.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoCuantica",
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 superposition_factor: float = 0.5,
                 entanglement_strength: float = 0.3,
                 decoherence_rate: float = 0.1,
                 measurement_probability: float = 0.7,
                 quantum_temperature: float = 0.1,
                 interference_factor: float = 0.2,
                 quantum_noise: float = 0.05):
        """
        Inicializa la neurona de refuerzo cuántica.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            superposition_factor: Factor de superposición cuántica
            entanglement_strength: Fuerza del entrelazamiento
            decoherence_rate: Tasa de decoherencia
            measurement_probability: Probabilidad de medición cuántica
            quantum_temperature: Temperatura cuántica
            interference_factor: Factor de interferencia cuántica
            quantum_noise: Ruido cuántico
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.superposition_factor = superposition_factor
        self.entanglement_strength = entanglement_strength
        self.decoherence_rate = decoherence_rate
        self.measurement_probability = measurement_probability
        self.quantum_temperature = quantum_temperature
        self.interference_factor = interference_factor
        self.quantum_noise = quantum_noise

        # Pesos cuánticos (representación compleja)
        self.pesos_cuanticos_real = None
        self.pesos_cuanticos_imag = None
        self.sesgo_cuantico_real = None
        self.sesgo_cuantico_imag = None

        # Pesos clásicos (para compatibilidad)
        self.pesos_clasicos = None
        self.sesgo_clasico = None

        # Estados cuánticos
        self.estado_superposicion = None
        self.entrelazamiento_matrix = None
        self.fase_cuantica = None

        # Estadísticas específicas cuánticas
        self.estadisticas_cuanticas = {
            'superposicion_media': 0.0,
            'entrelazamiento_medio': 0.0,
            'decoherencia_actual': 0.0,
            'interferencia_constructiva': 0.0,
            'interferencia_destructiva': 0.0,
            'mediciones_cuanticas': 0,
            'ruido_cuantico_medio': 0.0,
            'estabilidad_cuantica': 0.0,
            'eficiencia_cuantica': 0.0,
            'coherencia_cuantica': 0.0
        }

        # Historial cuántico
        self.historial_superposicion = deque(maxlen=1000)
        self.historial_entrelazamiento = deque(maxlen=1000)
        self.historial_decoherencia = deque(maxlen=1000)
        self.historial_interferencia = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoCuantica creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos cuánticos híbridos.
        """
        # Pesos cuánticos con inicialización específica
        self.pesos_cuanticos_real = inicializar_pesos_cuanticos(
            (self.input_size, self.output_size),
            self.superposition_factor
        )
        self.pesos_cuanticos_imag = np.random.normal(0, 0.1, (self.input_size, self.output_size))

        # Sesgos cuánticos
        self.sesgo_cuantico_real = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
        self.sesgo_cuantico_imag = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Pesos clásicos (proyección de los cuánticos)
        self.pesos_clasicos = self.pesos_cuanticos_real.copy()
        self.sesgo_clasico = self.sesgo_cuantico_real.copy()

        # Inicializar estados cuánticos
        self.estado_superposicion = np.random.uniform(0, 2*np.pi, (self.input_size, self.output_size))
        self.entrelazamiento_matrix = np.random.uniform(-1, 1, (self.input_size, self.input_size))
        self.fase_cuantica = np.random.uniform(0, 2*np.pi, (self.input_size, self.output_size))

        logger.info("Pesos cuánticos híbridos inicializados")

    def _aplicar_superposicion_cuantica(self, pesos: np.ndarray) -> np.ndarray:
        """
        Aplica superposición cuántica a los pesos.

        Args:
            pesos: Pesos base

        Returns:
            Pesos con superposición cuántica
        """
        # Crear superposición de estados
        amplitud_0 = np.cos(self.estado_superposicion)
        amplitud_1 = np.sin(self.estado_superposicion)

        # Combinar amplitudes
        pesos_superposicion = amplitud_0 * pesos + amplitud_1 * self.pesos_cuanticos_real

        return pesos_superposicion

    def _aplicar_entrelazamiento_cuantico(self, pesos: np.ndarray) -> np.ndarray:
        """
        Aplica entrelazamiento cuántico entre pesos.

        Args:
            pesos: Pesos a entrelazar

        Returns:
            Pesos con entrelazamiento cuántico
        """
        # Aplicar matriz de entrelazamiento
        pesos_entrelazados = np.dot(self.entrelazamiento_matrix, pesos)

        # Escalar por fuerza de entrelazamiento
        pesos_entrelazados = (1 - self.entanglement_strength) * pesos + \
            self.entanglement_strength * pesos_entrelazados

        return pesos_entrelazados

    def _aplicar_interferencia_cuantica(self, pesos1: np.ndarray, pesos2: np.ndarray) -> np.ndarray:
        """
        Aplica interferencia cuántica entre dos conjuntos de pesos.

        Args:
            pesos1: Primer conjunto de pesos
            pesos2: Segundo conjunto de pesos

        Returns:
            Pesos con interferencia cuántica
        """
        # Calcular diferencia de fase
        diferencia_fase = self.fase_cuantica

        # Interferencia constructiva y destructiva
        interferencia_constructiva = np.cos(diferencia_fase)
        interferencia_destructiva = np.sin(diferencia_fase)

        # Combinar interferencias
        pesos_interferencia = (pesos1 * interferencia_constructiva +
                               pesos2 * interferencia_destructiva) * self.interference_factor

        return pesos_interferencia

    def _aplicar_decoherencia(self, pesos: np.ndarray) -> np.ndarray:
        """
        Aplica decoherencia cuántica a los pesos.

        Args:
            pesos: Pesos cuánticos

        Returns:
            Pesos con decoherencia aplicada
        """
        # Ruido cuántico
        ruido_cuantico = np.random.normal(0, self.quantum_noise, pesos.shape)

        # Aplicar decoherencia
        pesos_decoherencia = pesos * (1 - self.decoherence_rate) + ruido_cuantico * self.decoherence_rate

        return pesos_decoherencia

    def _medicion_cuantica(self, pesos: np.ndarray) -> np.ndarray:
        """
        Realiza medición cuántica de los pesos.

        Args:
            pesos: Pesos cuánticos

        Returns:
            Pesos después de la medición cuántica
        """
        # Probabilidad de medición
        if np.random.random() < self.measurement_probability:
            # Medición cuántica - colapsar a estado clásico
            pesos_medidos = np.real(pesos)  # Tomar parte real

            # Actualizar estadísticas
            self.estadisticas_cuanticas['mediciones_cuanticas'] += 1

            return pesos_medidos
        else:
            # Mantener estado cuántico
            return pesos

    def _calcular_estabilidad_cuantica(self) -> float:
        """
        Calcula la estabilidad cuántica del sistema.

        Returns:
            Score de estabilidad cuántica
        """
        if len(self.historial_superposicion) < 10:
            return 0.0

        # Estabilidad basada en consistencia de superposición
        superposiciones_recientes = np.array(self.historial_superposicion[-100:])
        varianza_superposicion = np.var(superposiciones_recientes)

        # Estabilidad basada en coherencia cuántica
        coherencia = 1.0 / (1.0 + self.estadisticas_cuanticas['decoherencia_actual'])

        # Combinar métricas
        estabilidad = (1.0 / (1.0 + varianza_superposicion) + coherencia) / 2.0

        self.estadisticas_cuanticas['estabilidad_cuantica'] = estabilidad
        self.estadisticas_cuanticas['coherencia_cuantica'] = coherencia

        return estabilidad

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante con pesos cuánticos híbridos.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado con pesos cuánticos
        """
        if self.pesos_cuanticos_real is None:
            self.inicializar_pesos()

        # Aplicar superposición cuántica
        pesos_superposicion = self._aplicar_superposicion_cuantica(self.pesos_cuanticos_real)

        # Aplicar entrelazamiento cuántico
        pesos_entrelazados = self._aplicar_entrelazamiento_cuantico(pesos_superposicion)

        # Aplicar interferencia cuántica
        pesos_interferencia = self._aplicar_interferencia_cuantica(
            pesos_entrelazados, self.pesos_cuanticos_imag
        )

        # Aplicar decoherencia
        pesos_decoherencia = self._aplicar_decoherencia(pesos_interferencia)

        # Medición cuántica
        pesos_finales = self._medicion_cuantica(pesos_decoherencia)

        # Calcular salida
        logits = np.dot(estado, pesos_finales) + self.sesgo_clasico

        # Aplicar temperatura cuántica
        logits = logits / self.quantum_temperature

        # Aplicar función de activación
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Actualizar estadísticas cuánticas
        self._actualizar_estadisticas_cuanticas()

        return probabilidades

    def _actualizar_estadisticas_cuanticas(self) -> None:
        """
        Actualiza las estadísticas cuánticas.
        """
        # Calcular métricas cuánticas
        superposicion_media = np.mean(np.abs(self.estado_superposicion))
        entrelazamiento_medio = np.mean(np.abs(self.entrelazamiento_matrix))
        ruido_cuantico_medio = np.mean(np.abs(np.random.normal(0, self.quantum_noise, (10, 10))))

        # Actualizar estadísticas
        self.estadisticas_cuanticas['superposicion_media'] = superposicion_media
        self.estadisticas_cuanticas['entrelazamiento_medio'] = entrelazamiento_medio
        self.estadisticas_cuanticas['decoherencia_actual'] = self.decoherence_rate
        self.estadisticas_cuanticas['ruido_cuantico_medio'] = ruido_cuantico_medio

        # Calcular interferencias
        interferencia_constructiva = np.mean(np.cos(self.fase_cuantica))
        interferencia_destructiva = np.mean(np.sin(self.fase_cuantica))

        self.estadisticas_cuanticas['interferencia_constructiva'] = interferencia_constructiva
        self.estadisticas_cuanticas['interferencia_destructiva'] = interferencia_destructiva

        # Guardar en historiales
        self.historial_superposicion.append(superposicion_media)
        self.historial_entrelazamiento.append(entrelazamiento_medio)
        self.historial_decoherencia.append(self.decoherence_rate)
        self.historial_interferencia.append(interferencia_constructiva)

        # Calcular estabilidad cuántica
        self._calcular_estabilidad_cuantica()

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción usando la política cuántica híbrida.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades cuánticas
        accion = np.random.choice(self.output_size, p=probabilidades[0])

        return accion

    def calcular_gradiente_cuantico(self, estado: np.ndarray, accion: int, recompensa: float,
                                    contexto: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula gradientes cuánticos para los pesos.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            contexto: Contexto adicional (opcional)

        Returns:
            Tupla con (gradiente_pesos, gradiente_sesgo)
        """
        if self.pesos_cuanticos_real is None:
            raise ValueError("Pesos cuánticos no inicializados")

        # Calcular probabilidades actuales
        probabilidades = self.forward(estado, contexto)

        # Calcular gradiente de política cuántica
        grad_log_prob = np.zeros(self.output_size)
        grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

        # Aplicar recompensa como ventaja
        grad_output = grad_log_prob * recompensa

        # Calcular gradientes cuánticos
        grad_pesos = np.outer(estado, grad_output)
        grad_sesgo = grad_output

        # Aplicar efectos cuánticos al gradiente
        grad_pesos = grad_pesos * (1 + self.superposition_factor * np.sin(self.fase_cuantica[0, :]))

        return grad_pesos, grad_sesgo

    def actualizar_pesos_cuanticos(self, gradiente_pesos: np.ndarray, gradiente_sesgo: np.ndarray) -> None:
        """
        Actualiza los pesos cuánticos híbridos.

        Args:
            gradiente_pesos: Gradiente de los pesos
            gradiente_sesgo: Gradiente del sesgo
        """
        if self.pesos_cuanticos_real is None:
            raise ValueError("Pesos cuánticos no inicializados")

        # Actualizar pesos cuánticos reales
        self.pesos_cuanticos_real -= self.learning_rate * gradiente_pesos
        self.sesgo_cuantico_real -= self.learning_rate * gradiente_sesgo

        # Actualizar pesos clásicos
        self.pesos_clasicos = self.pesos_cuanticos_real.copy()
        self.sesgo_clasico = self.sesgo_cuantico_real.copy()

        # Actualizar estados cuánticos
        self.estado_superposicion += self.learning_rate * np.random.uniform(-0.1, 0.1, self.estado_superposicion.shape)
        self.fase_cuantica += self.learning_rate * np.random.uniform(-0.1, 0.1, self.fase_cuantica.shape)

        # Guardar en historial de pesos
        self.historial_pesos.append(self.pesos_cuanticos_real.copy())

        # Calcular eficiencia cuántica
        self._calcular_eficiencia_cuantica()

    def _calcular_eficiencia_cuantica(self) -> None:
        """
        Calcula la eficiencia cuántica del sistema.
        """
        if len(self.historial_pesos) < 10:
            return

        # Eficiencia basada en estabilidad cuántica
        estabilidad = self.estadisticas_cuanticas['estabilidad_cuantica']

        # Eficiencia basada en coherencia cuántica
        coherencia = self.estadisticas_cuanticas['coherencia_cuantica']

        # Combinar métricas
        eficiencia = (estabilidad + coherencia) / 2.0

        self.estadisticas_cuanticas['eficiencia_cuantica'] = eficiencia

    def obtener_estadisticas_cuanticas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de la neurona cuántica.

        Returns:
            Diccionario con estadísticas cuánticas
        """
        if self.pesos_cuanticos_real is None:
            return {'estado': 'no_inicializada'}

        stats_cuanticas = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'superposition_factor': self.superposition_factor,
            'entanglement_strength': self.entanglement_strength,
            'decoherence_rate': self.decoherence_rate,
            'measurement_probability': self.measurement_probability,
            'quantum_temperature': self.quantum_temperature,
            'interference_factor': self.interference_factor,
            'quantum_noise': self.quantum_noise,
            'superposicion_media': self.estadisticas_cuanticas['superposicion_media'],
            'entrelazamiento_medio': self.estadisticas_cuanticas['entrelazamiento_medio'],
            'decoherencia_actual': self.estadisticas_cuanticas['decoherencia_actual'],
            'interferencia_constructiva': self.estadisticas_cuanticas['interferencia_constructiva'],
            'interferencia_destructiva': self.estadisticas_cuanticas['interferencia_destructiva'],
            'mediciones_cuanticas': self.estadisticas_cuanticas['mediciones_cuanticas'],
            'ruido_cuantico_medio': self.estadisticas_cuanticas['ruido_cuantico_medio'],
            'estabilidad_cuantica': self.estadisticas_cuanticas['estabilidad_cuantica'],
            'eficiencia_cuantica': self.estadisticas_cuanticas['eficiencia_cuantica'],
            'coherencia_cuantica': self.estadisticas_cuanticas['coherencia_cuantica']
        }

        return stats_cuanticas

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona cuántica.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas cuánticas
        estabilidad['estabilidad_cuantica_ok'] = self.estadisticas_cuanticas['estabilidad_cuantica'] > 0.7
        estabilidad['coherencia_cuantica_ok'] = self.estadisticas_cuanticas['coherencia_cuantica'] > 0.6
        estabilidad['eficiencia_cuantica_ok'] = self.estadisticas_cuanticas['eficiencia_cuantica'] > 0.5
        estabilidad['superposicion_balanceada'] = 0.1 < self.estadisticas_cuanticas['superposicion_media'] < 2.0
        estabilidad['entrelazamiento_apropiado'] = 0.1 < self.estadisticas_cuanticas['entrelazamiento_medio'] < 1.0

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     superposition_factor: float = None,
                                     entanglement_strength: float = None,
                                     decoherence_rate: float = None,
                                     measurement_probability: float = None,
                                     quantum_temperature: float = None,
                                     interference_factor: float = None,
                                     quantum_noise: float = None) -> None:
        """
        Reinicializa la neurona cuántica con nuevos parámetros.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if superposition_factor is not None:
            self.superposition_factor = superposition_factor
        if entanglement_strength is not None:
            self.entanglement_strength = entanglement_strength
        if decoherence_rate is not None:
            self.decoherence_rate = decoherence_rate
        if measurement_probability is not None:
            self.measurement_probability = measurement_probability
        if quantum_temperature is not None:
            self.quantum_temperature = quantum_temperature
        if interference_factor is not None:
            self.interference_factor = interference_factor
        if quantum_noise is not None:
            self.quantum_noise = quantum_noise

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar historiales específicos
        self.historial_superposicion.clear()
        self.historial_entrelazamiento.clear()
        self.historial_decoherencia.clear()
        self.historial_interferencia.clear()

        # Resetear estadísticas cuánticas
        for key in self.estadisticas_cuanticas:
            self.estadisticas_cuanticas[key] = 0.0

        logger.info(f"Neurona cuántica reinicializada: lr={self.learning_rate}, "
                    f"superposition={self.superposition_factor}, entanglement={self.entanglement_strength}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoCuantica(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"superposition={self.superposition_factor}, entanglement={self.entanglement_strength})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_6


def crear_neurona_cuantica(input_size: int, output_size: int,
                           configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoCuantica:
    """
    Función de conveniencia para crear una neurona cuántica.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona cuántica configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoCuantica(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoCuantica'),
        learning_rate=configuracion.get('learning_rate', LUCIA_ADVANCED_RL_CONFIG['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma']),
        superposition_factor=configuracion.get('superposition_factor', LUCIA_ADVANCED_RL_CONFIG['quantum_superposition_factor']),
        entanglement_strength=configuracion.get('entanglement_strength', 0.3),
        decoherence_rate=configuracion.get('decoherence_rate', 0.1),
        measurement_probability=configuracion.get('measurement_probability', 0.7),
        quantum_temperature=configuracion.get('quantum_temperature', 0.1),
        interference_factor=configuracion.get('interference_factor', 0.2),
        quantum_noise=configuracion.get('quantum_noise', 0.05)
    )


# Configuración específica para RFEN2_RN_6
RFEN2_RN_6_CONFIG = {
    'inicializacion_preferida': 'cuantica',
    'learning_rate_default': 0.0001,
    'gamma_default': 0.99,
    'superposition_factor_default': 0.5,
    'entanglement_strength_default': 0.3,
    'decoherence_rate_default': 0.1,
    'measurement_probability_default': 0.7,
    'quantum_temperature_default': 0.1,
    'interference_factor_default': 0.2,
    'quantum_noise_default': 0.05,
    'umbral_estabilidad_cuantica': 0.7,
    'umbral_coherencia_cuantica': 0.6,
    'umbral_eficiencia_cuantica': 0.5,
    'umbral_superposicion_balanceada': [0.1, 2.0],
    'umbral_entrelazamiento_apropiado': [0.1, 1.0]
}

logger.info("RFEN2_RN_6.py cargado correctamente - Neurona de Refuerzo Cuántica")
