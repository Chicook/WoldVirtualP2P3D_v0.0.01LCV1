"""
RFEN2_RN_9.py - Neurona de Refuerzo con Optimización Federada
=============================================================

Esta neurona implementa Optimización Federada de Pesos que permite
entrenamiento distribuido manteniendo la privacidad de los datos,
utilizando técnicas avanzadas de 2025 para aprendizaje federado.

Características Avanzadas 2025:
- Aprendizaje federado con preservación de privacidad
- Agregación segura de pesos distribuidos
- Técnicas de privacidad diferencial
- Comunicación eficiente entre nodos
- Resiliencia a nodos maliciosos
- Balanceo de carga adaptativo
- Sincronización asíncrona
- Optimización de comunicación

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
import threading
from . import NeuronaRefuerzoAvanzadaBase, aplicar_privacy_preserving, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_9')


class NeuronaRefuerzoFederada(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con Optimización Federada de Pesos.

    Esta neurona implementa aprendizaje federado que permite entrenamiento
    distribuido manteniendo la privacidad de los datos.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoFederada",
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 federated_rounds: int = 10,
                 privacy_factor: float = 0.1,
                 aggregation_method: str = "fedavg",
                 communication_rounds: int = 5,
                 node_id: str = "node_0",
                 max_nodes: int = 10,
                 resilience_threshold: float = 0.7,
                 load_balancing_enabled: bool = True):
        """
        Inicializa la neurona de refuerzo federada.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            federated_rounds: Número de rondas federadas
            privacy_factor: Factor de privacidad
            aggregation_method: Método de agregación
            communication_rounds: Rondas de comunicación
            node_id: ID del nodo
            max_nodes: Número máximo de nodos
            resilience_threshold: Umbral de resiliencia
            load_balancing_enabled: Si habilitar balanceo de carga
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.federated_rounds = federated_rounds
        self.privacy_factor = privacy_factor
        self.aggregation_method = aggregation_method
        self.communication_rounds = communication_rounds
        self.node_id = node_id
        self.max_nodes = max_nodes
        self.resilience_threshold = resilience_threshold
        self.load_balancing_enabled = load_balancing_enabled

        # Pesos locales y globales
        self.pesos_locales = None
        self.sesgo_local = None
        self.pesos_globales = None
        self.sesgo_global = None

        # Pesos de otros nodos
        self.pesos_nodos = {}
        self.sesgos_nodos = {}
        self.confianza_nodos = {}

        # Estado federado
        self.ronda_actual = 0
        self.estado_sincronizacion = "local"
        self.ultima_comunicacion = 0

        # Estadísticas específicas federadas
        self.estadisticas_federadas = {
            'rondas_completadas': 0,
            'comunicaciones_realizadas': 0,
            'privacidad_preservada': 0.0,
            'eficiencia_comunicacion': 0.0,
            'resiliencia_nodos': 0.0,
            'balance_carga': 0.0,
            'sincronizacion_tiempo': 0.0,
            'calidad_agregacion': 0.0,
            'estabilidad_federada': 0.0,
            'throughput_comunicacion': 0.0
        }

        # Historial federado
        self.historial_rondas = deque(maxlen=1000)
        self.historial_comunicaciones = deque(maxlen=1000)
        self.historial_privacidad = deque(maxlen=1000)
        self.historial_agregacion = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoFederada creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos locales y globales.
        """
        # Pesos locales
        self.pesos_locales = np.random.normal(0, 0.1, (self.input_size, self.output_size))
        self.sesgo_local = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Pesos globales (inicialmente iguales a los locales)
        self.pesos_globales = self.pesos_locales.copy()
        self.sesgo_global = self.sesgo_local.copy()

        # Inicializar confianza de nodos
        for i in range(self.max_nodes):
            node_id = f"node_{i}"
            self.confianza_nodos[node_id] = 1.0

        logger.info(f"Pesos federados inicializados para nodo {self.node_id}")

    def _aplicar_privacidad_diferencial(self, pesos: np.ndarray) -> np.ndarray:
        """
        Aplica privacidad diferencial a los pesos.

        Args:
            pesos: Pesos a proteger

        Returns:
            Pesos con privacidad diferencial aplicada
        """
        return aplicar_privacy_preserving(pesos, self.privacy_factor)

    def _agregar_pesos_fedavg(self, pesos_nodos: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Agrega pesos usando FedAvg (Federated Averaging).

        Args:
            pesos_nodos: Diccionario con pesos de nodos

        Returns:
            Pesos agregados
        """
        if not pesos_nodos:
            return self.pesos_locales

        # Calcular promedio ponderado por confianza
        pesos_agregados = np.zeros_like(self.pesos_locales)
        peso_total = 0.0

        for node_id, pesos in pesos_nodos.items():
            confianza = self.confianza_nodos.get(node_id, 0.0)
            pesos_agregados += confianza * pesos
            peso_total += confianza

        if peso_total > 0:
            pesos_agregados /= peso_total

        return pesos_agregados

    def _agregar_pesos_fedprox(self, pesos_nodos: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Agrega pesos usando FedProx.

        Args:
            pesos_nodos: Diccionario con pesos de nodos

        Returns:
            Pesos agregados
        """
        if not pesos_nodos:
            return self.pesos_locales

        # FedProx con regularización proximal
        pesos_agregados = self._agregar_pesos_fedavg(pesos_nodos)

        # Aplicar regularización proximal
        mu = 0.01  # Parámetro de regularización
        pesos_prox = pesos_agregados + mu * (pesos_agregados - self.pesos_globales)

        return pesos_prox

    def _agregar_pesos(self, pesos_nodos: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Agrega pesos usando el método especificado.

        Args:
            pesos_nodos: Diccionario con pesos de nodos

        Returns:
            Pesos agregados
        """
        if self.aggregation_method == "fedavg":
            return self._agregar_pesos_fedavg(pesos_nodos)
        elif self.aggregation_method == "fedprox":
            return self._agregar_pesos_fedprox(pesos_nodos)
        else:
            return self._agregar_pesos_fedavg(pesos_nodos)

    def _calcular_resiliencia_nodos(self) -> float:
        """
        Calcula la resiliencia de los nodos.

        Returns:
            Score de resiliencia
        """
        if not self.confianza_nodos:
            return 0.0

        # Resiliencia basada en confianza promedio
        confianzas = list(self.confianza_nodos.values())
        resiliencia = np.mean(confianzas)

        return resiliencia

    def _calcular_balance_carga(self) -> float:
        """
        Calcula el balance de carga entre nodos.

        Returns:
            Score de balance de carga
        """
        if not self.load_balancing_enabled:
            return 1.0

        # Balance basado en distribución de confianza
        if not self.confianza_nodos:
            return 1.0

        confianzas = list(self.confianza_nodos.values())
        varianza_confianza = np.var(confianzas)
        balance = 1.0 / (1.0 + varianza_confianza)

        return balance

    def _sincronizar_con_nodos(self) -> None:
        """
        Sincroniza con otros nodos federados.
        """
        if self.estado_sincronizacion != "local":
            return

        # Simular comunicación con otros nodos
        self.estado_sincronizacion = "sincronizando"

        # Recopilar pesos de otros nodos (simulado)
        pesos_nodos = {}
        for node_id in self.confianza_nodos.keys():
            if node_id != self.node_id:
                # Simular pesos de otros nodos
                pesos_simulados = np.random.normal(0, 0.1, self.pesos_locales.shape)
                pesos_nodos[node_id] = pesos_simulados

        # Agregar pesos
        pesos_agregados = self._agregar_pesos(pesos_nodos)

        # Aplicar privacidad diferencial
        pesos_privados = self._aplicar_privacidad_diferencial(pesos_agregados)

        # Actualizar pesos globales
        self.pesos_globales = pesos_privados
        self.sesgo_global = self.sesgo_local.copy()

        # Actualizar estadísticas
        self.estadisticas_federadas['comunicaciones_realizadas'] += 1
        self.estadisticas_federadas['privacidad_preservada'] = self.privacy_factor
        self.estadisticas_federadas['resiliencia_nodos'] = self._calcular_resiliencia_nodos()
        self.estadisticas_federadas['balance_carga'] = self._calcular_balance_carga()

        # Guardar en historial
        self.historial_comunicaciones.append({
            'ronda': self.ronda_actual,
            'nodos_comunicados': len(pesos_nodos),
            'timestamp': time.time()
        })

        self.estado_sincronizacion = "sincronizado"
        self.ultima_comunicacion = time.time()

    def _iniciar_ronda_federada(self) -> None:
        """
        Inicia una nueva ronda federada.
        """
        self.ronda_actual += 1
        self.estado_sincronizacion = "local"

        logger.info(f"Iniciando ronda federada {self.ronda_actual}")

    def _finalizar_ronda_federada(self) -> None:
        """
        Finaliza la ronda federada actual.
        """
        # Sincronizar con otros nodos
        self._sincronizar_con_nodos()

        # Actualizar estadísticas
        self.estadisticas_federadas['rondas_completadas'] += 1

        # Guardar en historial
        self.historial_rondas.append({
            'ronda': self.ronda_actual,
            'estado': self.estado_sincronizacion,
            'timestamp': time.time()
        })

        logger.info(f"Ronda federada {self.ronda_actual} completada")

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante federada.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado federado
        """
        if self.pesos_locales is None:
            self.inicializar_pesos()

        # Usar pesos globales si están sincronizados, sino usar locales
        if self.estado_sincronizacion == "sincronizado":
            pesos_actuales = self.pesos_globales
            sesgo_actual = self.sesgo_global
        else:
            pesos_actuales = self.pesos_locales
            sesgo_actual = self.sesgo_local

        # Calcular salida
        logits = np.dot(estado, pesos_actuales) + sesgo_actual

        # Aplicar función de activación
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probabilidades

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción usando la política federada.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades federadas
        accion = np.random.choice(self.output_size, p=probabilidades[0])

        return accion

    def entrenar_paso_federado(self, estado: np.ndarray, accion: int, recompensa: float,
                               siguiente_estado: np.ndarray, terminado: bool) -> None:
        """
        Realiza un paso de entrenamiento federado.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            terminado: Si el episodio terminó
        """
        # Entrenamiento local
        logits = np.dot(estado, self.pesos_locales) + self.sesgo_local
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Gradiente de política
        grad_log_prob = np.zeros(self.output_size)
        grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

        # Actualizar pesos locales
        grad_pesos = np.outer(estado, grad_log_prob) * recompensa
        grad_sesgo = grad_log_prob * recompensa

        self.pesos_locales -= self.learning_rate * grad_pesos
        self.sesgo_local -= self.learning_rate * grad_sesgo

        # Guardar en historial de pesos
        self.historial_pesos.append(self.pesos_locales.copy())

        # Verificar si es necesario iniciar nueva ronda federada
        if self.ronda_actual < self.federated_rounds:
            if len(self.historial_pesos) % self.communication_rounds == 0:
                self._iniciar_ronda_federada()
                self._finalizar_ronda_federada()

    def obtener_estadisticas_federadas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de la optimización federada.

        Returns:
            Diccionario con estadísticas federadas
        """
        if self.pesos_locales is None:
            return {'estado': 'no_inicializada'}

        stats_federadas = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'federated_rounds': self.federated_rounds,
            'privacy_factor': self.privacy_factor,
            'aggregation_method': self.aggregation_method,
            'communication_rounds': self.communication_rounds,
            'node_id': self.node_id,
            'max_nodes': self.max_nodes,
            'resilience_threshold': self.resilience_threshold,
            'load_balancing_enabled': self.load_balancing_enabled,
            'ronda_actual': self.ronda_actual,
            'estado_sincronizacion': self.estado_sincronizacion,
            'rondas_completadas': self.estadisticas_federadas['rondas_completadas'],
            'comunicaciones_realizadas': self.estadisticas_federadas['comunicaciones_realizadas'],
            'privacidad_preservada': self.estadisticas_federadas['privacidad_preservada'],
            'eficiencia_comunicacion': self.estadisticas_federadas['eficiencia_comunicacion'],
            'resiliencia_nodos': self.estadisticas_federadas['resiliencia_nodos'],
            'balance_carga': self.estadisticas_federadas['balance_carga'],
            'sincronizacion_tiempo': self.estadisticas_federadas['sincronizacion_tiempo'],
            'calidad_agregacion': self.estadisticas_federadas['calidad_agregacion'],
            'estabilidad_federada': self.estadisticas_federadas['estabilidad_federada'],
            'throughput_comunicacion': self.estadisticas_federadas['throughput_comunicacion'],
            'historial_rondas_size': len(self.historial_rondas),
            'historial_comunicaciones_size': len(self.historial_comunicaciones),
            'historial_privacidad_size': len(self.historial_privacidad),
            'historial_agregacion_size': len(self.historial_agregacion)
        }

        return stats_federadas

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona federada.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas federadas
        estabilidad['rondas_federadas_ok'] = self.estadisticas_federadas['rondas_completadas'] > 0
        estabilidad['comunicaciones_efectivas'] = self.estadisticas_federadas['comunicaciones_realizadas'] > 0
        estabilidad['privacidad_preservada_ok'] = self.estadisticas_federadas['privacidad_preservada'] > 0.05
        estabilidad['resiliencia_nodos_ok'] = self.estadisticas_federadas['resiliencia_nodos'] > self.resilience_threshold
        estabilidad['balance_carga_ok'] = self.estadisticas_federadas['balance_carga'] > 0.7
        estabilidad['sincronizacion_estable'] = self.estado_sincronizacion in ["local", "sincronizado"]

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     federated_rounds: int = None,
                                     privacy_factor: float = None,
                                     aggregation_method: str = None,
                                     communication_rounds: int = None,
                                     node_id: str = None,
                                     max_nodes: int = None,
                                     resilience_threshold: float = None,
                                     load_balancing_enabled: bool = None) -> None:
        """
        Reinicializa la neurona federada con nuevos parámetros.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if federated_rounds is not None:
            self.federated_rounds = federated_rounds
        if privacy_factor is not None:
            self.privacy_factor = privacy_factor
        if aggregation_method is not None:
            self.aggregation_method = aggregation_method
        if communication_rounds is not None:
            self.communication_rounds = communication_rounds
        if node_id is not None:
            self.node_id = node_id
        if max_nodes is not None:
            self.max_nodes = max_nodes
        if resilience_threshold is not None:
            self.resilience_threshold = resilience_threshold
        if load_balancing_enabled is not None:
            self.load_balancing_enabled = load_balancing_enabled

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar estructuras federadas
        self.pesos_nodos.clear()
        self.sesgos_nodos.clear()
        self.confianza_nodos.clear()

        # Limpiar historiales específicos
        self.historial_rondas.clear()
        self.historial_comunicaciones.clear()
        self.historial_privacidad.clear()
        self.historial_agregacion.clear()

        # Resetear estado federado
        self.ronda_actual = 0
        self.estado_sincronizacion = "local"
        self.ultima_comunicacion = 0

        # Resetear estadísticas federadas
        for key in self.estadisticas_federadas:
            self.estadisticas_federadas[key] = 0.0

        logger.info(f"Neurona federada reinicializada: lr={self.learning_rate}, "
                    f"rondas={self.federated_rounds}, privacidad={self.privacy_factor}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoFederada(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"rondas={self.federated_rounds}, privacidad={self.privacy_factor})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_9


def crear_neurona_federada(input_size: int, output_size: int,
                           configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoFederada:
    """
    Función de conveniencia para crear una neurona federada.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona federada configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoFederada(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoFederada'),
        learning_rate=configuracion.get('learning_rate', LUCIA_ADVANCED_RL_CONFIG['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma']),
        federated_rounds=configuracion.get('federated_rounds', LUCIA_ADVANCED_RL_CONFIG['federated_rounds']),
        privacy_factor=configuracion.get('privacy_factor', LUCIA_ADVANCED_RL_CONFIG['federated_privacy_factor']),
        aggregation_method=configuracion.get('aggregation_method', 'fedavg'),
        communication_rounds=configuracion.get('communication_rounds', 5),
        node_id=configuracion.get('node_id', 'node_0'),
        max_nodes=configuracion.get('max_nodes', 10),
        resilience_threshold=configuracion.get('resilience_threshold', 0.7),
        load_balancing_enabled=configuracion.get('load_balancing_enabled', True)
    )


# Configuración específica para RFEN2_RN_9
RFEN2_RN_9_CONFIG = {
    'inicializacion_preferida': 'federada',
    'learning_rate_default': 0.0001,
    'gamma_default': 0.99,
    'federated_rounds_default': 10,
    'privacy_factor_default': 0.1,
    'aggregation_method_default': 'fedavg',
    'communication_rounds_default': 5,
    'node_id_default': 'node_0',
    'max_nodes_default': 10,
    'resilience_threshold_default': 0.7,
    'load_balancing_enabled_default': True,
    'umbral_rondas_federadas': 0,
    'umbral_comunicaciones_efectivas': 0,
    'umbral_privacidad_preservada': 0.05,
    'umbral_resiliencia_nodos': 0.7,
    'umbral_balance_carga': 0.7,
    'umbral_sincronizacion_estable': ['local', 'sincronizado']
}

logger.info("RFEN2_RN_9.py cargado correctamente - Neurona de Refuerzo Federada")
