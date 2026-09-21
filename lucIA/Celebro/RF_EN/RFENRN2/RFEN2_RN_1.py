"""
RFEN2_RN_1.py - Neurona de Refuerzo con Red Neuronal Líquida
============================================================

Esta neurona implementa una Red Neuronal Líquida (Liquid Neural Network) con
pesos adaptativos dinámicos, inspirada en las investigaciones más recientes
de 2025 sobre redes neuronales líquidas.

Características Avanzadas 2025:
- Pesos adaptativos que cambian dinámicamente según el contexto
- Ecuaciones diferenciales que gobiernan el comportamiento temporal
- Aprendizaje continuo incluso después del entrenamiento inicial
- Adaptación automática a cambios en el entorno
- Consolidación de memoria durante estados líquidos
- Optimización de pesos basada en dinámicas fluidas

Autor: LucIA Development Team
Versión: 2.0.0
Fecha: 2025
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
import logging
from collections import deque
import threading
import time
from . import NeuronaRefuerzoAvanzadaBase, inicializar_pesos_liquidos, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_1')


class NeuronaRefuerzoLiquida(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con Red Neuronal Líquida (Liquid Neural Network).

    Esta neurona implementa pesos adaptativos dinámicos que cambian según
    el contexto y el tiempo, permitiendo aprendizaje continuo y adaptación
    automática a cambios en el entorno.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoLiquida",
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 liquid_dynamics_factor: float = 0.1,
                 adaptation_rate: float = 0.01,
                 memory_decay: float = 0.95,
                 temperature: float = 0.5,
                 viscosity: float = 0.1,
                 flow_rate: float = 0.05):
        """
        Inicializa la neurona de refuerzo líquida.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje base
            gamma: Factor de descuento
            liquid_dynamics_factor: Factor de dinámica líquida
            adaptation_rate: Tasa de adaptación de pesos
            memory_decay: Factor de decaimiento de memoria
            temperature: Temperatura del sistema líquido
            viscosity: Viscosidad del líquido neural
            flow_rate: Tasa de flujo de información
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.liquid_dynamics_factor = liquid_dynamics_factor
        self.adaptation_rate = adaptation_rate
        self.memory_decay = memory_decay
        self.temperature = temperature
        self.viscosity = viscosity
        self.flow_rate = flow_rate

        # Pesos líquidos adaptativos
        self.pesos_liquidos = None
        self.sesgo_liquido = None
        self.pesos_base = None
        self.sesgo_base = None

        # Estados líquidos dinámicos
        self.estado_liquido = np.zeros((input_size,), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
        self.velocidad_liquida = np.zeros((input_size,), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])
        self.presion_liquida = np.zeros((output_size,), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Memoria líquida adaptativa
        self.memoria_liquida = deque(maxlen=1000)
        self.contexto_liquido = deque(maxlen=100)
        self.historial_dinamico = deque(maxlen=500)

        # Estadísticas específicas de red líquida
        self.estadisticas_liquidas = {
            'adaptacion_dinamica': 0.0,
            'flujo_informacion': 0.0,
            'viscosidad_actual': 0.0,
            'temperatura_actual': 0.0,
            'presion_media': 0.0,
            'velocidad_media': 0.0,
            'estabilidad_liquida': 0.0,
            'eficiencia_adaptacion': 0.0,
            'memoria_consolidada': 0.0,
            'cambio_pesos_dinamico': 0.0
        }

        # Historial específico para análisis líquido
        self.historial_adaptacion = []
        self.historial_flujo = []
        self.historial_viscosidad = []
        self.historial_temperatura = []
        self.historial_presion = []
        self.historial_velocidad = []

        logger.info(f"NeuronaRefuerzoLiquida creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos líquidos adaptativos.
        """
        # Pesos base con inicialización líquida
        self.pesos_base = inicializar_pesos_liquidos(
            (self.input_size, self.output_size),
            self.liquid_dynamics_factor
        )
        self.sesgo_base = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Pesos líquidos adaptativos (inicialmente iguales a los base)
        self.pesos_liquidos = self.pesos_base.copy()
        self.sesgo_liquido = self.sesgo_base.copy()

        # Inicializar estados líquidos
        self.estado_liquido = np.random.normal(0, 0.1, self.input_size).astype(LUCIA_ADVANCED_RL_CONFIG['precision'])
        self.velocidad_liquida = np.random.normal(0, 0.05, self.input_size).astype(LUCIA_ADVANCED_RL_CONFIG['precision'])
        self.presion_liquida = np.random.normal(0, 0.1, self.output_size).astype(LUCIA_ADVANCED_RL_CONFIG['precision'])

        logger.info(f"Pesos líquidos inicializados con dinámica factor={self.liquid_dynamics_factor}")

    def _actualizar_dinamica_liquida(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> None:
        """
        Actualiza la dinámica líquida del sistema.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)
        """
        # Calcular fuerzas líquidas
        fuerza_entrada = estado - self.estado_liquido
        fuerza_viscosa = -self.viscosity * self.velocidad_liquida
        fuerza_presion = -self.presion_liquida * self.flow_rate

        # Actualizar velocidad líquida
        self.velocidad_liquida += self.liquid_dynamics_factor * (
            fuerza_entrada + fuerza_viscosa + fuerza_presion[:self.input_size]
        )

        # Actualizar estado líquido
        self.estado_liquido += self.velocidad_liquida * self.flow_rate

        # Actualizar presión líquida
        if contexto is not None:
            presion_contexto = np.dot(contexto, self.pesos_liquidos.T)
            self.presion_liquida += self.adaptation_rate * presion_contexto

        # Aplicar decaimiento de memoria
        self.velocidad_liquida *= self.memory_decay
        self.presion_liquida *= self.memory_decay

        # Guardar en historial dinámico
        self.historial_dinamico.append({
            'estado_liquido': self.estado_liquido.copy(),
            'velocidad_liquida': self.velocidad_liquida.copy(),
            'presion_liquida': self.presion_liquida.copy(),
            'timestamp': time.time()
        })

    def _adaptar_pesos_dinamicamente(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> None:
        """
        Adapta los pesos dinámicamente basado en el contexto líquido.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)
        """
        if contexto is None:
            contexto = self.estado_liquido

        # Calcular factor de adaptación basado en dinámica líquida
        velocidad_norm = np.linalg.norm(self.velocidad_liquida)
        presion_norm = np.linalg.norm(self.presion_liquida)

        adaptation_factor = 1.0 + self.adaptation_rate * (velocidad_norm + presion_norm)

        # Adaptar pesos líquidos
        self.pesos_liquidos = calcular_adaptacion_dinamica(self.pesos_liquidos, contexto)
        self.pesos_liquidos *= adaptation_factor

        # Aplicar consolidación de memoria
        if len(self.memoria_liquida) > 10:
            self.pesos_liquidos = aplicar_consolidacion_memoria(
                self.pesos_liquidos,
                [entry['pesos'] for entry in self.memoria_liquida]
            )

        # Actualizar estadísticas
        self.estadisticas_liquidas['adaptacion_dinamica'] = adaptation_factor
        self.estadisticas_liquidas['flujo_informacion'] = velocidad_norm
        self.estadisticas_liquidas['viscosidad_actual'] = self.viscosity
        self.estadisticas_liquidas['temperatura_actual'] = self.temperature
        self.estadisticas_liquidas['presion_media'] = presion_norm
        self.estadisticas_liquidas['velocidad_media'] = velocidad_norm

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante con dinámica líquida.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado con dinámica líquida
        """
        if self.pesos_liquidos is None:
            self.inicializar_pesos()

        # Asegurar que el estado sea 2D
        if estado.ndim == 1:
            estado = estado.reshape(1, -1)

        # Actualizar dinámica líquida
        self._actualizar_dinamica_liquida(estado, contexto)

        # Adaptar pesos dinámicamente
        self._adaptar_pesos_dinamicamente(estado, contexto)

        # Calcular salida con pesos líquidos adaptativos
        # Reshape de estado_liquido para broadcast correcto
        estado_liquido_2d = self.estado_liquido.reshape(1, -1)
        estado_liquido_adaptado = estado_liquido_2d + estado * self.flow_rate

        # Aplicar temperatura líquida
        logits = np.dot(estado_liquido_adaptado, self.pesos_liquidos) + self.sesgo_liquido
        logits = logits / self.temperature

        # Aplicar función de activación líquida (softmax con dinámica)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Guardar en memoria líquida
        self.memoria_liquida.append({
            'estado': estado.copy(),
            'contexto': contexto.copy() if contexto is not None else None,
            'pesos': self.pesos_liquidos.copy(),
            'probabilidades': probabilidades.copy(),
            'timestamp': time.time()
        })

        # Guardar contexto líquido
        if contexto is not None:
            self.contexto_liquido.append(contexto.copy())

        return probabilidades

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción usando la política líquida adaptativa.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades líquidas
        accion = np.random.choice(self.output_size, p=probabilidades[0])

        return accion

    def calcular_gradiente_liquido(self, estado: np.ndarray, accion: int,
                                   recompensa: float, siguiente_estado: np.ndarray,
                                   contexto: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula gradientes con dinámica líquida.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            contexto: Contexto adicional (opcional)

        Returns:
            Tupla con (gradiente_pesos, gradiente_sesgo)
        """
        if self.pesos_liquidos is None:
            raise ValueError("Pesos líquidos no inicializados")

        # Calcular probabilidades actuales
        probabilidades_actuales = self.forward(estado, contexto)

        # Calcular valor objetivo
        valor_objetivo = recompensa + self.gamma * np.max(self.forward(siguiente_estado, contexto))

        # Calcular gradiente de política líquida
        grad_log_prob = np.zeros(self.output_size)
        grad_log_prob[accion] = 1.0 / (probabilidades_actuales[0, accion] + 1e-8)

        # Aplicar dinámica líquida al gradiente
        estado_liquido_adaptado = self.estado_liquido + estado * self.flow_rate

        gradiente_pesos = np.outer(estado_liquido_adaptado, grad_log_prob) * valor_objetivo
        gradiente_sesgo = grad_log_prob * valor_objetivo

        # Aplicar factor de viscosidad al gradiente
        gradiente_pesos *= (1.0 - self.viscosity)
        gradiente_sesgo *= (1.0 - self.viscosity)

        return gradiente_pesos, gradiente_sesgo

    def actualizar_pesos_liquidos(self, gradiente_pesos: np.ndarray, gradiente_sesgo: np.ndarray,
                                  learning_rate: float = None, contexto: Optional[np.ndarray] = None) -> None:
        """
        Actualiza los pesos líquidos con dinámica adaptativa.

        Args:
            gradiente_pesos: Gradiente de los pesos
            gradiente_sesgo: Gradiente del sesgo
            learning_rate: Tasa de aprendizaje
            contexto: Contexto adicional (opcional)
        """
        if learning_rate is None:
            learning_rate = self.learning_rate

        # Aplicar tasa de aprendizaje adaptativa basada en dinámica líquida
        velocidad_norm = np.linalg.norm(self.velocidad_liquida)
        presion_norm = np.linalg.norm(self.presion_liquida)

        adaptive_lr = learning_rate * (1.0 + self.adaptation_rate * (velocidad_norm + presion_norm))

        # Actualizar pesos líquidos
        self.pesos_liquidos -= adaptive_lr * gradiente_pesos
        self.sesgo_liquido -= adaptive_lr * gradiente_sesgo

        # Actualizar pesos base con decaimiento
        self.pesos_base = self.pesos_base * self.memory_decay + self.pesos_liquidos * (1 - self.memory_decay)
        self.sesgo_base = self.sesgo_base * self.memory_decay + self.sesgo_liquido * (1 - self.memory_decay)

        # Guardar historial de pesos
        self.historial_pesos.append(self.pesos_liquidos.copy())

        # Actualizar estadísticas
        self.estadisticas_liquidas['cambio_pesos_dinamico'] = np.linalg.norm(gradiente_pesos)
        self.estadisticas_liquidas['eficiencia_adaptacion'] = adaptive_lr / learning_rate

        # Guardar en historiales específicos
        self.historial_adaptacion.append(self.estadisticas_liquidas['adaptacion_dinamica'])
        self.historial_flujo.append(self.estadisticas_liquidas['flujo_informacion'])
        self.historial_viscosidad.append(self.estadisticas_liquidas['viscosidad_actual'])
        self.historial_temperatura.append(self.estadisticas_liquidas['temperatura_actual'])
        self.historial_presion.append(self.estadisticas_liquidas['presion_media'])
        self.historial_velocidad.append(self.estadisticas_liquidas['velocidad_media'])

    def calcular_estabilidad_liquida(self) -> float:
        """
        Calcula la estabilidad del sistema líquido.

        Returns:
            Score de estabilidad líquida (0-1)
        """
        if len(self.historial_adaptacion) < 10:
            return 0.0

        # Calcular varianza de adaptación
        adaptacion_var = np.var(self.historial_adaptacion[-100:])

        # Calcular consistencia de flujo
        flujo_consistencia = 1.0 / (1.0 + np.var(self.historial_flujo[-100:]))

        # Calcular estabilidad de viscosidad
        viscosidad_estabilidad = 1.0 / (1.0 + np.var(self.historial_viscosidad[-100:]))

        # Combinar métricas
        estabilidad = (flujo_consistencia + viscosidad_estabilidad + (1.0 / (1.0 + adaptacion_var))) / 3.0

        self.estadisticas_liquidas['estabilidad_liquida'] = estabilidad
        return estabilidad

    def obtener_estadisticas_liquidas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de la red líquida.

        Returns:
            Diccionario con estadísticas líquidas
        """
        if self.pesos_liquidos is None:
            return {'estado': 'no_inicializada'}

        # Calcular estabilidad líquida
        estabilidad = self.calcular_estabilidad_liquida()

        stats_liquidas = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'liquid_dynamics_factor': self.liquid_dynamics_factor,
            'adaptation_rate': self.adaptation_rate,
            'memory_decay': self.memory_decay,
            'temperature': self.temperature,
            'viscosity': self.viscosity,
            'flow_rate': self.flow_rate,
            'adaptacion_dinamica': self.estadisticas_liquidas['adaptacion_dinamica'],
            'flujo_informacion': self.estadisticas_liquidas['flujo_informacion'],
            'viscosidad_actual': self.estadisticas_liquidas['viscosidad_actual'],
            'temperatura_actual': self.estadisticas_liquidas['temperatura_actual'],
            'presion_media': self.estadisticas_liquidas['presion_media'],
            'velocidad_media': self.estadisticas_liquidas['velocidad_media'],
            'estabilidad_liquida': estabilidad,
            'eficiencia_adaptacion': self.estadisticas_liquidas['eficiencia_adaptacion'],
            'cambio_pesos_dinamico': self.estadisticas_liquidas['cambio_pesos_dinamico'],
            'memoria_liquida_size': len(self.memoria_liquida),
            'contexto_liquido_size': len(self.contexto_liquido),
            'historial_dinamico_size': len(self.historial_dinamico)
        }

        return stats_liquidas

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona líquida.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas de red líquida
        estabilidad['dinamica_liquida_estable'] = self.estadisticas_liquidas['estabilidad_liquida'] > 0.7
        estabilidad['adaptacion_eficiente'] = self.estadisticas_liquidas['eficiencia_adaptacion'] < 2.0
        estabilidad['flujo_informacion_balanceado'] = 0.1 < self.estadisticas_liquidas['flujo_informacion'] < 1.0
        estabilidad['viscosidad_apropiada'] = 0.01 < self.viscosity < 0.5
        estabilidad['temperatura_estable'] = 0.1 < self.temperature < 2.0

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     liquid_dynamics_factor: float = None,
                                     adaptation_rate: float = None,
                                     memory_decay: float = None,
                                     temperature: float = None,
                                     viscosity: float = None,
                                     flow_rate: float = None) -> None:
        """
        Reinicializa la neurona líquida con nuevos parámetros.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if liquid_dynamics_factor is not None:
            self.liquid_dynamics_factor = liquid_dynamics_factor
        if adaptation_rate is not None:
            self.adaptation_rate = adaptation_rate
        if memory_decay is not None:
            self.memory_decay = memory_decay
        if temperature is not None:
            self.temperature = temperature
        if viscosity is not None:
            self.viscosity = viscosity
        if flow_rate is not None:
            self.flow_rate = flow_rate

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar historiales específicos
        self.historial_adaptacion.clear()
        self.historial_flujo.clear()
        self.historial_viscosidad.clear()
        self.historial_temperatura.clear()
        self.historial_presion.clear()
        self.historial_velocidad.clear()

        # Limpiar memorias líquidas
        self.memoria_liquida.clear()
        self.contexto_liquido.clear()
        self.historial_dinamico.clear()

        # Resetear estadísticas líquidas
        for key in self.estadisticas_liquidas:
            self.estadisticas_liquidas[key] = 0.0

        logger.info(f"Neurona líquida reinicializada: lr={self.learning_rate}, "
                    f"dinamica={self.liquid_dynamics_factor}, adaptacion={self.adaptation_rate}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoLiquida(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"dinamica={self.liquid_dynamics_factor}, adaptacion={self.adaptation_rate})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_1


def crear_neurona_liquida(input_size: int, output_size: int,
                          configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoLiquida:
    """
    Función de conveniencia para crear una neurona líquida.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona líquida configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoLiquida(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoLiquida'),
        learning_rate=configuracion.get('learning_rate', LUCIA_ADVANCED_RL_CONFIG['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma']),
        liquid_dynamics_factor=configuracion.get('liquid_dynamics_factor', LUCIA_ADVANCED_RL_CONFIG['liquid_dynamics_factor']),
        adaptation_rate=configuracion.get('adaptation_rate', 0.01),
        memory_decay=configuracion.get('memory_decay', 0.95),
        temperature=configuracion.get('temperature', LUCIA_ADVANCED_RL_CONFIG['default_temperature']),
        viscosity=configuracion.get('viscosity', 0.1),
        flow_rate=configuracion.get('flow_rate', 0.05)
    )


# Configuración específica para RFEN2_RN_1
RFEN2_RN_1_CONFIG = {
    'inicializacion_preferida': 'liquida',
    'learning_rate_default': 0.0001,
    'gamma_default': 0.99,
    'liquid_dynamics_factor_default': 0.1,
    'adaptation_rate_default': 0.01,
    'memory_decay_default': 0.95,
    'temperature_default': 0.5,
    'viscosity_default': 0.1,
    'flow_rate_default': 0.05,
    'umbral_estabilidad_liquida': 0.7,
    'umbral_adaptacion_eficiente': 2.0,
    'umbral_flujo_balanceado': [0.1, 1.0],
    'umbral_viscosidad_apropiada': [0.01, 0.5],
    'umbral_temperatura_estable': [0.1, 2.0]
}

logger.info("RFEN2_RN_1.py cargado correctamente - Neurona de Refuerzo Líquida")
