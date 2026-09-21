"""
RFEN2_RN_8.py - Neurona de Refuerzo Explicable (XAI)
====================================================

Esta neurona implementa Redes Neuronales Explicables (XAI) que proporcionan
interpretabilidad y transparencia en las decisiones, utilizando técnicas
avanzadas de 2025 para explicabilidad de IA.

Características Avanzadas 2025:
- Explicabilidad de decisiones en tiempo real
- Análisis de importancia de características
- Visualización de patrones de decisión
- Interpretabilidad de pesos y activaciones
- Métricas de confianza y certeza
- Análisis de sensibilidad de entrada
- Explicaciones naturales de decisiones
- Auditoría de sesgos y fairness

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
from . import NeuronaRefuerzoAvanzadaBase, calcular_explicabilidad, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_8')


class NeuronaRefuerzoExplicable(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con capacidades de explicabilidad (XAI).

    Esta neurona implementa técnicas de explicabilidad que proporcionan
    interpretabilidad y transparencia en las decisiones de IA.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoExplicable",
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 explicability_threshold: float = 0.8,
                 confidence_threshold: float = 0.7,
                 sensitivity_factor: float = 0.1,
                 interpretability_weight: float = 0.3,
                 transparency_factor: float = 0.5,
                 bias_detection_enabled: bool = True,
                 fairness_weight: float = 0.2):
        """
        Inicializa la neurona de refuerzo explicable.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            explicability_threshold: Umbral de explicabilidad
            confidence_threshold: Umbral de confianza
            sensitivity_factor: Factor de sensibilidad
            interpretability_weight: Peso de interpretabilidad
            transparency_factor: Factor de transparencia
            bias_detection_enabled: Si detectar sesgos
            fairness_weight: Peso de fairness
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.explicability_threshold = explicability_threshold
        self.confidence_threshold = confidence_threshold
        self.sensitivity_factor = sensitivity_factor
        self.interpretability_weight = interpretability_weight
        self.transparency_factor = transparency_factor
        self.bias_detection_enabled = bias_detection_enabled
        self.fairness_weight = fairness_weight

        # Pesos explicables
        self.pesos_explicables = None
        self.sesgo_explicable = None

        # Métricas de explicabilidad
        self.importancia_caracteristicas = np.zeros(input_size)
        self.confianza_acciones = np.zeros(output_size)
        self.sensibilidad_entrada = np.zeros(input_size)

        # Historial de explicaciones
        self.historial_explicaciones = deque(maxlen=1000)
        self.historial_decisiones = deque(maxlen=1000)
        self.historial_confianza = deque(maxlen=1000)

        # Estadísticas específicas de explicabilidad
        self.estadisticas_explicables = {
            'explicabilidad_media': 0.0,
            'confianza_media': 0.0,
            'transparencia_score': 0.0,
            'interpretabilidad_score': 0.0,
            'sensibilidad_media': 0.0,
            'bias_detectado': 0.0,
            'fairness_score': 0.0,
            'consistencia_decisiones': 0.0,
            'estabilidad_explicaciones': 0.0,
            'eficiencia_explicacion': 0.0
        }

        # Historial específico para análisis de explicabilidad
        self.historial_explicabilidad = deque(maxlen=1000)
        self.historial_transparencia = deque(maxlen=1000)
        self.historial_interpretabilidad = deque(maxlen=1000)
        self.historial_sensibilidad = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoExplicable creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos explicables.
        """
        # Pesos con inicialización que favorece la explicabilidad
        self.pesos_explicables = np.random.normal(0, 0.1, (self.input_size, self.output_size))
        self.sesgo_explicable = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Inicializar métricas de explicabilidad
        self.importancia_caracteristicas = np.ones(self.input_size) / self.input_size
        self.confianza_acciones = np.ones(self.output_size) / self.output_size
        self.sensibilidad_entrada = np.zeros(self.input_size)

        logger.info("Pesos explicables inicializados")

    def _calcular_importancia_caracteristicas(self, estado: np.ndarray) -> np.ndarray:
        """
        Calcula la importancia de cada característica de entrada.

        Args:
            estado: Estado actual

        Returns:
            Importancia de cada característica
        """
        if self.pesos_explicables is None:
            return np.zeros(self.input_size)

        # Calcular importancia basada en pesos
        importancia_pesos = np.sum(np.abs(self.pesos_explicables), axis=1)

        # Calcular importancia basada en activación
        importancia_activacion = np.abs(estado)

        # Combinar importancias
        importancia_total = (importancia_pesos + importancia_activacion) / 2.0

        # Normalizar
        if np.sum(importancia_total) > 0:
            importancia_total = importancia_total / np.sum(importancia_total)

        return importancia_total

    def _calcular_confianza_accion(self, probabilidades: np.ndarray) -> float:
        """
        Calcula la confianza en la acción seleccionada.

        Args:
            probabilidades: Probabilidades de acciones

        Returns:
            Confianza en la acción
        """
        # Confianza basada en la diferencia entre la mejor y segunda mejor acción
        probabilidades_ordenadas = np.sort(probabilidades[0])[::-1]

        if len(probabilidades_ordenadas) > 1:
            confianza = probabilidades_ordenadas[0] - probabilidades_ordenadas[1]
        else:
            confianza = probabilidades_ordenadas[0]

        return confianza

    def _calcular_sensibilidad_entrada(self, estado: np.ndarray) -> np.ndarray:
        """
        Calcula la sensibilidad de la salida a cambios en la entrada.

        Args:
            estado: Estado actual

        Returns:
            Sensibilidad de cada característica de entrada
        """
        if self.pesos_explicables is None:
            return np.zeros(self.input_size)

        # Calcular gradientes aproximados
        sensibilidad = np.sum(np.abs(self.pesos_explicables), axis=1)

        # Aplicar factor de sensibilidad
        sensibilidad = sensibilidad * self.sensitivity_factor

        return sensibilidad

    def _detectar_sesgos(self, estado: np.ndarray, accion: int) -> float:
        """
        Detecta sesgos en las decisiones.

        Args:
            estado: Estado actual
            accion: Acción tomada

        Returns:
            Score de sesgo detectado
        """
        if not self.bias_detection_enabled:
            return 0.0

        # Detectar sesgos basados en patrones históricos
        if len(self.historial_decisiones) < 10:
            return 0.0

        # Analizar distribución de acciones
        acciones_recientes = [d['accion'] for d in list(self.historial_decisiones)[-100:]]
        distribucion_acciones = np.bincount(acciones_recientes, minlength=self.output_size)

        # Calcular sesgo basado en desbalance
        distribucion_normalizada = distribucion_acciones / np.sum(distribucion_acciones)
        sesgo = np.var(distribucion_normalizada)

        return sesgo

    def _calcular_fairness_score(self, estado: np.ndarray, accion: int) -> float:
        """
        Calcula el score de fairness de la decisión.

        Args:
            estado: Estado actual
            accion: Acción tomada

        Returns:
            Score de fairness
        """
        # Fairness basado en consistencia de decisiones similares
        if len(self.historial_decisiones) < 10:
            return 1.0

        # Buscar decisiones similares
        decisiones_similares = []
        for decision in self.historial_decisiones:
            similitud = np.dot(estado, decision['estado']) / (
                np.linalg.norm(estado) * np.linalg.norm(decision['estado']) + 1e-8
            )
            if similitud > 0.8:
                decisiones_similares.append(decision['accion'])

        if len(decisiones_similares) > 0:
            # Calcular consistencia
            acciones_similares = np.array(decisiones_similares)
            consistencia = np.sum(acciones_similares == accion) / len(acciones_similares)
            fairness = consistencia
        else:
            fairness = 1.0

        return fairness

    def _generar_explicacion(self, estado: np.ndarray, accion: int,
                             probabilidades: np.ndarray) -> Dict[str, Any]:
        """
        Genera una explicación de la decisión tomada.

        Args:
            estado: Estado actual
            accion: Acción tomada
            probabilidades: Probabilidades de acciones

        Returns:
            Diccionario con explicación de la decisión
        """
        # Calcular métricas de explicabilidad
        importancia = self._calcular_importancia_caracteristicas(estado)
        confianza = self._calcular_confianza_accion(probabilidades)
        sensibilidad = self._calcular_sensibilidad_entrada(estado)
        sesgo = self._detectar_sesgos(estado, accion)
        fairness = self._calcular_fairness_score(estado, accion)

        # Generar explicación
        explicacion = {
            'accion_seleccionada': accion,
            'probabilidad_accion': probabilidades[0, accion],
            'confianza': confianza,
            'importancia_caracteristicas': importancia.tolist(),
            'sensibilidad_entrada': sensibilidad.tolist(),
            'sesgo_detectado': sesgo,
            'fairness_score': fairness,
            'explicabilidad_score': calcular_explicabilidad(self.pesos_explicables, estado),
            'timestamp': time.time()
        }

        return explicacion

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante con explicabilidad.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado con explicabilidad
        """
        if self.pesos_explicables is None:
            self.inicializar_pesos()

        # Calcular salida
        logits = np.dot(estado, self.pesos_explicables) + self.sesgo_explicable

        # Aplicar función de activación
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Actualizar métricas de explicabilidad
        self.importancia_caracteristicas = self._calcular_importancia_caracteristicas(estado)
        self.confianza_acciones = probabilidades[0]
        self.sensibilidad_entrada = self._calcular_sensibilidad_entrada(estado)

        return probabilidades

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción con explicabilidad.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades
        accion = np.random.choice(self.output_size, p=probabilidades[0])

        # Generar explicación
        explicacion = self._generar_explicacion(estado, accion, probabilidades)

        # Guardar en historial
        self.historial_explicaciones.append(explicacion)
        self.historial_decisiones.append({
            'estado': estado.copy(),
            'accion': accion,
            'probabilidades': probabilidades.copy(),
            'timestamp': time.time()
        })
        self.historial_confianza.append(explicacion['confianza'])

        # Actualizar estadísticas
        self._actualizar_estadisticas_explicables(explicacion)

        return accion

    def _actualizar_estadisticas_explicables(self, explicacion: Dict[str, Any]) -> None:
        """
        Actualiza las estadísticas de explicabilidad.
        """
        # Actualizar métricas principales
        self.estadisticas_explicables['explicabilidad_media'] = explicacion['explicabilidad_score']
        self.estadisticas_explicables['confianza_media'] = explicacion['confianza']
        self.estadisticas_explicables['transparencia_score'] = explicacion['fairness_score']
        self.estadisticas_explicables['interpretabilidad_score'] = np.mean(explicacion['importancia_caracteristicas'])
        self.estadisticas_explicables['sensibilidad_media'] = np.mean(explicacion['sensibilidad_entrada'])
        self.estadisticas_explicables['bias_detectado'] = explicacion['sesgo_detectado']
        self.estadisticas_explicables['fairness_score'] = explicacion['fairness_score']

        # Calcular consistencia de decisiones
        if len(self.historial_decisiones) > 10:
            decisiones_recientes = [d['accion'] for d in list(self.historial_decisiones)[-100:]]
            consistencia = 1.0 - np.var(decisiones_recientes) / (np.mean(decisiones_recientes) + 1e-8)
            self.estadisticas_explicables['consistencia_decisiones'] = consistencia

        # Calcular estabilidad de explicaciones
        if len(self.historial_explicaciones) > 10:
            explicabilidades_recientes = [e['explicabilidad_score'] for e in
                                          list(self.historial_explicaciones)[-100:]]
            estabilidad = 1.0 / (1.0 + np.var(explicabilidades_recientes))
            self.estadisticas_explicables['estabilidad_explicaciones'] = estabilidad

        # Guardar en historiales
        self.historial_explicabilidad.append(explicacion['explicabilidad_score'])
        self.historial_transparencia.append(explicacion['fairness_score'])
        self.historial_interpretabilidad.append(np.mean(explicacion['importancia_caracteristicas']))
        self.historial_sensibilidad.append(np.mean(explicacion['sensibilidad_entrada']))

    def obtener_explicacion_decision(self, estado: np.ndarray) -> Dict[str, Any]:
        """
        Obtiene una explicación detallada de la decisión para un estado.

        Args:
            estado: Estado para explicar

        Returns:
            Explicación detallada de la decisión
        """
        probabilidades = self.forward(estado)
        accion = np.argmax(probabilidades[0])

        explicacion = self._generar_explicacion(estado, accion, probabilidades)

        # Agregar información adicional
        explicacion['estado_entrada'] = estado.tolist()
        explicacion['probabilidades_todas'] = probabilidades[0].tolist()
        explicacion['razonamiento'] = f"Acción {accion} seleccionada con {explicacion['confianza']:.3f} de confianza"

        return explicacion

    def obtener_estadisticas_explicables(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de la explicabilidad.

        Returns:
            Diccionario con estadísticas de explicabilidad
        """
        if self.pesos_explicables is None:
            return {'estado': 'no_inicializada'}

        stats_explicables = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'explicability_threshold': self.explicability_threshold,
            'confidence_threshold': self.confidence_threshold,
            'sensitivity_factor': self.sensitivity_factor,
            'interpretability_weight': self.interpretability_weight,
            'transparency_factor': self.transparency_factor,
            'bias_detection_enabled': self.bias_detection_enabled,
            'fairness_weight': self.fairness_weight,
            'explicabilidad_media': self.estadisticas_explicables['explicabilidad_media'],
            'confianza_media': self.estadisticas_explicables['confianza_media'],
            'transparencia_score': self.estadisticas_explicables['transparencia_score'],
            'interpretabilidad_score': self.estadisticas_explicables['interpretabilidad_score'],
            'sensibilidad_media': self.estadisticas_explicables['sensibilidad_media'],
            'bias_detectado': self.estadisticas_explicables['bias_detectado'],
            'fairness_score': self.estadisticas_explicables['fairness_score'],
            'consistencia_decisiones': self.estadisticas_explicables['consistencia_decisiones'],
            'estabilidad_explicaciones': self.estadisticas_explicables['estabilidad_explicaciones'],
            'eficiencia_explicacion': self.estadisticas_explicables['eficiencia_explicacion'],
            'historial_explicaciones_size': len(self.historial_explicaciones),
            'historial_decisiones_size': len(self.historial_decisiones),
            'historial_confianza_size': len(self.historial_confianza)
        }

        return stats_explicables

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona explicable.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas de explicabilidad
        estabilidad['explicabilidad_suficiente'] = self.estadisticas_explicables['explicabilidad_media'] > self.explicability_threshold
        estabilidad['confianza_apropiada'] = self.estadisticas_explicables['confianza_media'] > self.confidence_threshold
        estabilidad['transparencia_ok'] = self.estadisticas_explicables['transparencia_score'] > 0.7
        estabilidad['interpretabilidad_ok'] = self.estadisticas_explicables['interpretabilidad_score'] > 0.5
        estabilidad['bias_controlado'] = self.estadisticas_explicables['bias_detectado'] < 0.5
        estabilidad['fairness_ok'] = self.estadisticas_explicables['fairness_score'] > 0.7
        estabilidad['consistencia_decisiones_ok'] = self.estadisticas_explicables['consistencia_decisiones'] > 0.6
        estabilidad['estabilidad_explicaciones_ok'] = self.estadisticas_explicables['estabilidad_explicaciones'] > 0.7

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     explicability_threshold: float = None,
                                     confidence_threshold: float = None,
                                     sensitivity_factor: float = None,
                                     interpretability_weight: float = None,
                                     transparency_factor: float = None,
                                     bias_detection_enabled: bool = None,
                                     fairness_weight: float = None) -> None:
        """
        Reinicializa la neurona explicable con nuevos parámetros.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if explicability_threshold is not None:
            self.explicability_threshold = explicability_threshold
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if sensitivity_factor is not None:
            self.sensitivity_factor = sensitivity_factor
        if interpretability_weight is not None:
            self.interpretability_weight = interpretability_weight
        if transparency_factor is not None:
            self.transparency_factor = transparency_factor
        if bias_detection_enabled is not None:
            self.bias_detection_enabled = bias_detection_enabled
        if fairness_weight is not None:
            self.fairness_weight = fairness_weight

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar historiales específicos
        self.historial_explicaciones.clear()
        self.historial_decisiones.clear()
        self.historial_confianza.clear()
        self.historial_explicabilidad.clear()
        self.historial_transparencia.clear()
        self.historial_interpretabilidad.clear()
        self.historial_sensibilidad.clear()

        # Resetear estadísticas explicables
        for key in self.estadisticas_explicables:
            self.estadisticas_explicables[key] = 0.0

        # Resetear métricas
        self.importancia_caracteristicas = np.zeros(self.input_size)
        self.confianza_acciones = np.zeros(self.output_size)
        self.sensibilidad_entrada = np.zeros(self.input_size)

        logger.info(f"Neurona explicable reinicializada: lr={self.learning_rate}, "
                    f"explicability_threshold={self.explicability_threshold}, "
                    f"confidence_threshold={self.confidence_threshold}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoExplicable(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"explicability_threshold={self.explicability_threshold}, "
                f"confidence_threshold={self.confidence_threshold})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_8


def crear_neurona_explicable(input_size: int, output_size: int,
                             configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoExplicable:
    """
    Función de conveniencia para crear una neurona explicable.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona explicable configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoExplicable(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoExplicable'),
        learning_rate=configuracion.get('learning_rate', LUCIA_ADVANCED_RL_CONFIG['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma']),
        explicability_threshold=configuracion.get('explicability_threshold', LUCIA_ADVANCED_RL_CONFIG['explainability_threshold']),
        confidence_threshold=configuracion.get('confidence_threshold', 0.7),
        sensitivity_factor=configuracion.get('sensitivity_factor', 0.1),
        interpretability_weight=configuracion.get('interpretability_weight', 0.3),
        transparency_factor=configuracion.get('transparency_factor', 0.5),
        bias_detection_enabled=configuracion.get('bias_detection_enabled', True),
        fairness_weight=configuracion.get('fairness_weight', 0.2)
    )


# Configuración específica para RFEN2_RN_8
RFEN2_RN_8_CONFIG = {
    'inicializacion_preferida': 'explicable',
    'learning_rate_default': 0.0001,
    'gamma_default': 0.99,
    'explicability_threshold_default': 0.8,
    'confidence_threshold_default': 0.7,
    'sensitivity_factor_default': 0.1,
    'interpretability_weight_default': 0.3,
    'transparency_factor_default': 0.5,
    'bias_detection_enabled_default': True,
    'fairness_weight_default': 0.2,
    'umbral_explicabilidad_suficiente': 0.8,
    'umbral_confianza_apropiada': 0.7,
    'umbral_transparencia': 0.7,
    'umbral_interpretabilidad': 0.5,
    'umbral_bias_controlado': 0.5,
    'umbral_fairness': 0.7,
    'umbral_consistencia_decisiones': 0.6,
    'umbral_estabilidad_explicaciones': 0.7
}

logger.info("RFEN2_RN_8.py cargado correctamente - Neurona de Refuerzo Explicable")
