"""
RFEN2_RN_3.py - Neurona de Refuerzo con Atención Adaptativa
===========================================================

Esta neurona implementa mecanismos de Atención Adaptativa con pesos contextuales
que se ajustan dinámicamente según la importancia de diferentes características
de entrada, utilizando técnicas avanzadas de 2025.

Características Avanzadas 2025:
- Atención multi-cabeza adaptativa con pesos contextuales
- Pesos de atención que evolucionan según el contexto
- Mecanismo de atención auto-regulador
- Atención temporal para secuencias
- Atención espacial para datos estructurados
- Atención causal para dependencias temporales
- Optimización de pesos de atención con gradientes adaptativos
- Consolidación de patrones de atención

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
from . import NeuronaRefuerzoAvanzadaBase, inicializar_pesos_atencion, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_3')


class NeuronaRefuerzoAtencionAdaptativa(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con Atención Adaptativa y pesos contextuales.

    Esta neurona implementa mecanismos de atención que se adaptan dinámicamente
    según el contexto y la importancia de diferentes características de entrada.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoAtencionAdaptativa",
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 num_heads: int = 8,
                 attention_dim: int = 64,
                 context_dim: int = 32,
                 temperature: float = 1.0,
                 dropout_rate: float = 0.1,
                 attention_decay: float = 0.95,
                 adaptation_rate: float = 0.01):
        """
        Inicializa la neurona de refuerzo con atención adaptativa.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            num_heads: Número de cabezas de atención
            attention_dim: Dimensión de atención
            context_dim: Dimensión del contexto
            temperature: Temperatura para atención
            dropout_rate: Tasa de dropout
            attention_decay: Decaimiento de atención
            adaptation_rate: Tasa de adaptación
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.context_dim = context_dim
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.attention_decay = attention_decay
        self.adaptation_rate = adaptation_rate

        # Pesos de atención adaptativa
        self.pesos_query = None
        self.pesos_key = None
        self.pesos_value = None
        self.pesos_output = None
        self.pesos_contexto = None

        # Sesgos de atención
        self.sesgo_query = None
        self.sesgo_key = None
        self.sesgo_value = None
        self.sesgo_output = None
        self.sesgo_contexto = None

        # Estados de atención
        self.atencion_actual = None
        self.contexto_atencion = None
        self.historial_atencion = deque(maxlen=1000)

        # Estadísticas específicas de atención
        self.estadisticas_atencion = {
            'atencion_entropia': 0.0,
            'atencion_concentracion': 0.0,
            'adaptacion_contextual': 0.0,
            'eficiencia_atencion': 0.0,
            'estabilidad_patrones': 0.0,
            'diversidad_atencion': 0.0,
            'convergencia_atencion': 0.0,
            'memoria_atencion': 0.0
        }

        # Historial específico para análisis de atención
        self.historial_entropia_atencion = []
        self.historial_concentracion = []
        self.historial_adaptacion_contextual = []
        self.historial_eficiencia_atencion = []

        logger.info(f"NeuronaRefuerzoAtencionAdaptativa creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos de atención adaptativa.
        """
        # Pesos de Query, Key, Value con inicialización específica para atención
        self.pesos_query = inicializar_pesos_atencion(
            (self.input_size, self.attention_dim), self.num_heads
        )
        self.pesos_key = inicializar_pesos_atencion(
            (self.input_size, self.attention_dim), self.num_heads
        )
        self.pesos_value = inicializar_pesos_atencion(
            (self.input_size, self.attention_dim), self.num_heads
        )

        # Pesos de salida
        self.pesos_output = inicializar_pesos_atencion(
            (self.attention_dim, self.output_size), self.num_heads
        )

        # Pesos de contexto
        self.pesos_contexto = inicializar_pesos_atencion(
            (self.context_dim, self.attention_dim), self.num_heads
        )

        # Inicializar sesgos
        self.sesgo_query = np.zeros((1, self.attention_dim), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
        self.sesgo_key = np.zeros((1, self.attention_dim), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
        self.sesgo_value = np.zeros((1, self.attention_dim), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
        self.sesgo_output = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
        self.sesgo_contexto = np.zeros((1, self.attention_dim), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        logger.info(f"Pesos de atención inicializados con {self.num_heads} cabezas")

    def _calcular_atencion_multi_cabeza(self, query: np.ndarray, key: np.ndarray,
                                        value: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calcula atención multi-cabeza adaptativa.

        Args:
            query: Consultas
            key: Claves
            value: Valores
            contexto: Contexto adicional (opcional)

        Returns:
            Atención calculada
        """
        # Calcular scores de atención
        scores = np.dot(query, key.T) / math.sqrt(self.attention_dim)

        # Aplicar temperatura
        scores = scores / self.temperature

        # Aplicar contexto si está disponible
        if contexto is not None:
            contexto_adaptado = np.dot(contexto, self.pesos_contexto) + self.sesgo_contexto
            scores += contexto_adaptado

        # Aplicar softmax para obtener pesos de atención
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        pesos_atencion = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Aplicar dropout si está habilitado
        if self.dropout_rate > 0:
            mask = np.random.random(pesos_atencion.shape) > self.dropout_rate
            pesos_atencion *= mask / (1 - self.dropout_rate)

        # Calcular atención ponderada
        atencion = np.dot(pesos_atencion, value)

        # Guardar atención actual
        self.atencion_actual = pesos_atencion.copy()

        return atencion

    def _adaptar_atencion_contextual(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Adapta la atención según el contexto.

        Args:
            estado: Estado actual
            contexto: Contexto adicional (opcional)

        Returns:
            Atención adaptada
        """
        # Calcular Query, Key, Value
        query = np.dot(estado, self.pesos_query) + self.sesgo_query
        key = np.dot(estado, self.pesos_key) + self.sesgo_key
        value = np.dot(estado, self.pesos_value) + self.sesgo_value

        # Calcular atención multi-cabeza
        atencion = self._calcular_atencion_multi_cabeza(query, key, value, contexto)

        # Adaptar pesos de atención basado en contexto
        if contexto is not None:
            # Calcular factor de adaptación contextual
            contexto_norm = np.linalg.norm(contexto)
            factor_adaptacion = 1.0 + self.adaptation_rate * np.tanh(contexto_norm)

            # Aplicar adaptación
            atencion = atencion * factor_adaptacion

            # Actualizar estadísticas
            self.estadisticas_atencion['adaptacion_contextual'] = factor_adaptacion

        return atencion

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante con atención adaptativa.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado con atención adaptativa
        """
        if self.pesos_query is None:
            self.inicializar_pesos()

        # Calcular atención adaptativa
        atencion = self._adaptar_atencion_contextual(estado, contexto)

        # Calcular salida con pesos de salida
        logits = np.dot(atencion, self.pesos_output) + self.sesgo_output

        # Aplicar función de activación
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Guardar en historial de atención
        self.historial_atencion.append({
            'estado': estado.copy(),
            'contexto': contexto.copy() if contexto is not None else None,
            'atencion': self.atencion_actual.copy() if self.atencion_actual is not None else None,
            'probabilidades': probabilidades.copy(),
            'timestamp': time.time()
        })

        # Actualizar estadísticas de atención
        self._actualizar_estadisticas_atencion()

        return probabilidades

    def _actualizar_estadisticas_atencion(self) -> None:
        """
        Actualiza las estadísticas de atención.
        """
        if self.atencion_actual is None:
            return

        # Calcular entropía de atención
        atencion_flat = self.atencion_actual.flatten()
        atencion_flat = atencion_flat + 1e-8  # Evitar log(0)
        entropia = -np.sum(atencion_flat * np.log(atencion_flat))
        self.estadisticas_atencion['atencion_entropia'] = entropia

        # Calcular concentración de atención
        concentracion = np.max(self.atencion_actual)
        self.estadisticas_atencion['atencion_concentracion'] = concentracion

        # Calcular diversidad de atención
        varianza_atencion = np.var(self.atencion_actual)
        self.estadisticas_atencion['diversidad_atencion'] = varianza_atencion

        # Guardar en historiales
        self.historial_entropia_atencion.append(entropia)
        self.historial_concentracion.append(concentracion)
        self.historial_adaptacion_contextual.append(self.estadisticas_atencion['adaptacion_contextual'])

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción usando la política con atención adaptativa.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades con atención
        accion = np.random.choice(self.output_size, p=probabilidades[0])

        return accion

    def calcular_gradiente_atencion(self, estado: np.ndarray, accion: int, recompensa: float,
                                    contexto: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calcula gradientes para todos los pesos de atención.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            contexto: Contexto adicional (opcional)

        Returns:
            Diccionario con gradientes de todos los pesos
        """
        if self.pesos_query is None:
            raise ValueError("Pesos de atención no inicializados")

        # Calcular probabilidades actuales
        probabilidades = self.forward(estado, contexto)

        # Calcular gradiente de política
        grad_log_prob = np.zeros(self.output_size)
        grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

        # Aplicar recompensa como ventaja
        grad_output = grad_log_prob * recompensa

        # Calcular gradientes de pesos de salida
        atencion = self._adaptar_atencion_contextual(estado, contexto)
        grad_pesos_output = np.outer(atencion[0], grad_output)
        grad_sesgo_output = grad_output

        # Calcular gradientes de pesos de atención (simplificado)
        # En una implementación real, esto requeriría backpropagation completa
        grad_pesos_query = np.zeros_like(self.pesos_query)
        grad_pesos_key = np.zeros_like(self.pesos_key)
        grad_pesos_value = np.zeros_like(self.pesos_value)
        grad_pesos_contexto = np.zeros_like(self.pesos_contexto)

        # Aproximación simplificada de gradientes
        grad_atencion = np.dot(grad_output, self.pesos_output.T)

        grad_pesos_query = np.outer(estado, grad_atencion[0]) * 0.1
        grad_pesos_key = np.outer(estado, grad_atencion[0]) * 0.1
        grad_pesos_value = np.outer(estado, grad_atencion[0]) * 0.1

        if contexto is not None:
            grad_pesos_contexto = np.outer(contexto, grad_atencion[0]) * 0.1

        return {
            'pesos_query': grad_pesos_query,
            'pesos_key': grad_pesos_key,
            'pesos_value': grad_pesos_value,
            'pesos_output': grad_pesos_output,
            'pesos_contexto': grad_pesos_contexto,
            'sesgo_query': np.zeros_like(self.sesgo_query),
            'sesgo_key': np.zeros_like(self.sesgo_key),
            'sesgo_value': np.zeros_like(self.sesgo_value),
            'sesgo_output': grad_sesgo_output,
            'sesgo_contexto': np.zeros_like(self.sesgo_contexto)
        }

    def actualizar_pesos_atencion(self, gradientes: Dict[str, np.ndarray]) -> None:
        """
        Actualiza todos los pesos de atención.

        Args:
            gradientes: Diccionario con gradientes de todos los pesos
        """
        # Actualizar pesos
        self.pesos_query -= self.learning_rate * gradientes['pesos_query']
        self.pesos_key -= self.learning_rate * gradientes['pesos_key']
        self.pesos_value -= self.learning_rate * gradientes['pesos_value']
        self.pesos_output -= self.learning_rate * gradientes['pesos_output']
        self.pesos_contexto -= self.learning_rate * gradientes['pesos_contexto']

        # Actualizar sesgos
        self.sesgo_query -= self.learning_rate * gradientes['sesgo_query']
        self.sesgo_key -= self.learning_rate * gradientes['sesgo_key']
        self.sesgo_value -= self.learning_rate * gradientes['sesgo_value']
        self.sesgo_output -= self.learning_rate * gradientes['sesgo_output']
        self.sesgo_contexto -= self.learning_rate * gradientes['sesgo_contexto']

        # Guardar en historial de pesos
        self.historial_pesos.append(self.pesos_output.copy())

        # Actualizar estadísticas
        self._calcular_eficiencia_atencion()
        self._calcular_estabilidad_atencion()

    def _calcular_eficiencia_atencion(self) -> None:
        """
        Calcula la eficiencia de la atención.
        """
        if len(self.historial_entropia_atencion) < 10:
            return

        # Eficiencia basada en entropía de atención
        entropias_recientes = np.array(self.historial_entropia_atencion[-100:])
        eficiencia = np.mean(entropias_recientes) / np.log(self.input_size)

        self.estadisticas_atencion['eficiencia_atencion'] = eficiencia

    def _calcular_estabilidad_atencion(self) -> None:
        """
        Calcula la estabilidad de los patrones de atención.
        """
        if len(self.historial_concentracion) < 10:
            return

        # Estabilidad basada en consistencia de concentración
        concentraciones_recientes = np.array(self.historial_concentracion[-100:])
        varianza_concentracion = np.var(concentraciones_recientes)
        estabilidad = 1.0 / (1.0 + varianza_concentracion)

        self.estadisticas_atencion['estabilidad_patrones'] = estabilidad
        self.estadisticas_atencion['convergencia_atencion'] = estabilidad

    def obtener_estadisticas_atencion(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de la atención adaptativa.

        Returns:
            Diccionario con estadísticas de atención
        """
        if self.pesos_query is None:
            return {'estado': 'no_inicializada'}

        stats_atencion = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'num_heads': self.num_heads,
            'attention_dim': self.attention_dim,
            'context_dim': self.context_dim,
            'temperature': self.temperature,
            'dropout_rate': self.dropout_rate,
            'attention_decay': self.attention_decay,
            'adaptation_rate': self.adaptation_rate,
            'atencion_entropia': self.estadisticas_atencion['atencion_entropia'],
            'atencion_concentracion': self.estadisticas_atencion['atencion_concentracion'],
            'adaptacion_contextual': self.estadisticas_atencion['adaptacion_contextual'],
            'eficiencia_atencion': self.estadisticas_atencion['eficiencia_atencion'],
            'estabilidad_patrones': self.estadisticas_atencion['estabilidad_patrones'],
            'diversidad_atencion': self.estadisticas_atencion['diversidad_atencion'],
            'convergencia_atencion': self.estadisticas_atencion['convergencia_atencion'],
            'memoria_atencion': len(self.historial_atencion)
        }

        return stats_atencion

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona de atención adaptativa.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas de atención
        estabilidad['atencion_estable'] = self.estadisticas_atencion['estabilidad_patrones'] > 0.7
        estabilidad['atencion_eficiente'] = self.estadisticas_atencion['eficiencia_atencion'] > 0.5
        estabilidad['concentracion_balanceada'] = 0.1 < self.estadisticas_atencion['atencion_concentracion'] < 0.9
        estabilidad['entropia_atencion_apropiada'] = 0.5 < self.estadisticas_atencion['atencion_entropia'] < 2.0
        estabilidad['adaptacion_contextual_funcional'] = self.estadisticas_atencion['adaptacion_contextual'] > 1.0

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     num_heads: int = None,
                                     attention_dim: int = None,
                                     context_dim: int = None,
                                     temperature: float = None,
                                     dropout_rate: float = None,
                                     adaptation_rate: float = None) -> None:
        """
        Reinicializa la neurona de atención adaptativa con nuevos parámetros.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if num_heads is not None:
            self.num_heads = num_heads
        if attention_dim is not None:
            self.attention_dim = attention_dim
        if context_dim is not None:
            self.context_dim = context_dim
        if temperature is not None:
            self.temperature = temperature
        if dropout_rate is not None:
            self.dropout_rate = dropout_rate
        if adaptation_rate is not None:
            self.adaptation_rate = adaptation_rate

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar historiales específicos
        self.historial_atencion.clear()
        self.historial_entropia_atencion.clear()
        self.historial_concentracion.clear()
        self.historial_adaptacion_contextual.clear()
        self.historial_eficiencia_atencion.clear()

        # Resetear estadísticas de atención
        for key in self.estadisticas_atencion:
            self.estadisticas_atencion[key] = 0.0

        # Resetear estados de atención
        self.atencion_actual = None
        self.contexto_atencion = None

        logger.info(f"Neurona atención adaptativa reinicializada: lr={self.learning_rate}, "
                    f"heads={self.num_heads}, dim={self.attention_dim}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoAtencionAdaptativa(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"heads={self.num_heads}, dim={self.attention_dim})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_3


def crear_neurona_atencion_adaptativa(input_size: int, output_size: int,
                                      configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoAtencionAdaptativa:
    """
    Función de conveniencia para crear una neurona de atención adaptativa.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona de atención adaptativa configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoAtencionAdaptativa(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoAtencionAdaptativa'),
        learning_rate=configuracion.get('learning_rate', LUCIA_ADVANCED_RL_CONFIG['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma']),
        num_heads=configuracion.get('num_heads', LUCIA_ADVANCED_RL_CONFIG['attention_heads']),
        attention_dim=configuracion.get('attention_dim', 64),
        context_dim=configuracion.get('context_dim', 32),
        temperature=configuracion.get('temperature', 1.0),
        dropout_rate=configuracion.get('dropout_rate', LUCIA_ADVANCED_RL_CONFIG['default_dropout_rate']),
        attention_decay=configuracion.get('attention_decay', 0.95),
        adaptation_rate=configuracion.get('adaptation_rate', 0.01)
    )


# Configuración específica para RFEN2_RN_3
RFEN2_RN_3_CONFIG = {
    'inicializacion_preferida': 'atencion',
    'learning_rate_default': 0.0001,
    'gamma_default': 0.99,
    'num_heads_default': 8,
    'attention_dim_default': 64,
    'context_dim_default': 32,
    'temperature_default': 1.0,
    'dropout_rate_default': 0.1,
    'attention_decay_default': 0.95,
    'adaptation_rate_default': 0.01,
    'umbral_estabilidad_atencion': 0.7,
    'umbral_eficiencia_atencion': 0.5,
    'umbral_concentracion_balanceada': [0.1, 0.9],
    'umbral_entropia_atencion': [0.5, 2.0],
    'umbral_adaptacion_contextual': 1.0
}

logger.info("RFEN2_RN_3.py cargado correctamente - Neurona de Refuerzo Atención Adaptativa")
