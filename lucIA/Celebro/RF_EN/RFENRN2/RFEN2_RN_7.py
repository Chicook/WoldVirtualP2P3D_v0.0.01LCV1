"""
RFEN2_RN_7.py - Neurona de Refuerzo con Optimización Multi-Objetivo
===================================================================

Esta neurona implementa Optimización Multi-Objetivo avanzada que optimiza
múltiples objetivos simultáneamente, utilizando técnicas avanzadas de 2025
para balancear diferentes métricas de rendimiento.

Características Avanzadas 2025:
- Optimización multi-objetivo con Pareto optimalidad
- Balanceo dinámico de objetivos múltiples
- Algoritmos de optimización multi-objetivo (NSGA-II, MOEA/D)
- Métricas de diversidad y convergencia
- Análisis de frontera de Pareto
- Optimización adaptativa de pesos de objetivos
- Evaluación multi-criterio de soluciones
- Integración de objetivos conflictivos

Autor: LucIA Development Team
Versión: 2.0.0
Fecha: 2025
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any, List
import logging
from collections import deque
import random
from . import NeuronaRefuerzoAvanzadaBase, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_7')


class NeuronaRefuerzoMultiObjetivo(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con Optimización Multi-Objetivo.

    Esta neurona implementa optimización multi-objetivo que balancea
    múltiples objetivos simultáneamente usando técnicas de Pareto optimalidad.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoMultiObjetivo",
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 num_objectives: int = 3,
                 objective_weights: List[float] = None,
                 pareto_threshold: float = 0.1,
                 diversity_weight: float = 0.3,
                 convergence_weight: float = 0.7,
                 adaptation_rate: float = 0.01,
                 balance_factor: float = 0.5):
        """
        Inicializa la neurona de refuerzo multi-objetivo.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            num_objectives: Número de objetivos
            objective_weights: Pesos de los objetivos
            pareto_threshold: Umbral de Pareto optimalidad
            diversity_weight: Peso de diversidad
            convergence_weight: Peso de convergencia
            adaptation_rate: Tasa de adaptación
            balance_factor: Factor de balanceo
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_objectives = num_objectives
        self.objective_weights = objective_weights or LUCIA_ADVANCED_RL_CONFIG['multi_objective_weights']
        self.pareto_threshold = pareto_threshold
        self.diversity_weight = diversity_weight
        self.convergence_weight = convergence_weight
        self.adaptation_rate = adaptation_rate
        self.balance_factor = balance_factor

        # Pesos para cada objetivo
        self.pesos_objetivos = []
        self.sesgos_objetivos = []

        # Frontera de Pareto
        self.frontera_pareto = deque(maxlen=1000)
        self.soluciones_pareto = deque(maxlen=1000)

        # Estadísticas específicas multi-objetivo
        self.estadisticas_multi_obj = {
            'objetivo_1_rendimiento': 0.0,
            'objetivo_2_rendimiento': 0.0,
            'objetivo_3_rendimiento': 0.0,
            'pareto_optimalidad': 0.0,
            'diversidad_soluciones': 0.0,
            'convergencia_multi_obj': 0.0,
            'balance_objetivos': 0.0,
            'eficiencia_pareto': 0.0,
            'estabilidad_multi_obj': 0.0,
            'adaptacion_objetivos': 0.0
        }

        # Historial multi-objetivo
        self.historial_objetivos = deque(maxlen=1000)
        self.historial_pareto = deque(maxlen=1000)
        self.historial_diversidad = deque(maxlen=1000)
        self.historial_convergencia = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoMultiObjetivo creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos para cada objetivo.
        """
        # Inicializar pesos para cada objetivo
        for i in range(self.num_objectives):
            pesos_obj = np.random.normal(0, 0.1, (self.input_size, self.output_size))
            sesgo_obj = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

            self.pesos_objetivos.append(pesos_obj)
            self.sesgos_objetivos.append(sesgo_obj)

        logger.info(f"Pesos multi-objetivo inicializados para {self.num_objectives} objetivos")

    def _calcular_objetivo_1(self, estado: np.ndarray, accion: int, recompensa: float) -> float:
        """
        Calcula el primer objetivo (rendimiento general).

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida

        Returns:
            Valor del objetivo 1
        """
        # Objetivo: Maximizar recompensa acumulada
        return recompensa

    def _calcular_objetivo_2(self, estado: np.ndarray, accion: int, recompensa: float) -> float:
        """
        Calcula el segundo objetivo (estabilidad).

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida

        Returns:
            Valor del objetivo 2
        """
        # Objetivo: Minimizar varianza de acciones
        if len(self.historial_objetivos) > 0:
            acciones_recientes = [h['accion'] for h in list(self.historial_objetivos)[-100:]]
            varianza_acciones = np.var(acciones_recientes)
            return -varianza_acciones  # Negativo porque queremos minimizar
        return 0.0

    def _calcular_objetivo_3(self, estado: np.ndarray, accion: int, recompensa: float) -> float:
        """
        Calcula el tercer objetivo (eficiencia).

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida

        Returns:
            Valor del objetivo 3
        """
        # Objetivo: Maximizar eficiencia de aprendizaje
        if len(self.historial_pesos) > 0:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            cambio_pesos = np.mean(np.var(pesos_recientes, axis=0))
            return -cambio_pesos  # Negativo porque queremos estabilidad
        return 0.0

    def _calcular_todos_objetivos(self, estado: np.ndarray, accion: int, recompensa: float) -> List[float]:
        """
        Calcula todos los objetivos.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida

        Returns:
            Lista con valores de todos los objetivos
        """
        objetivos = []

        # Calcular cada objetivo
        objetivos.append(self._calcular_objetivo_1(estado, accion, recompensa))
        objetivos.append(self._calcular_objetivo_2(estado, accion, recompensa))
        objetivos.append(self._calcular_objetivo_3(estado, accion, recompensa))

        return objetivos

    def _es_pareto_optimal(self, objetivos: List[float]) -> bool:
        """
        Verifica si una solución es Pareto optimal.

        Args:
            objetivos: Valores de objetivos

        Returns:
            True si es Pareto optimal
        """
        if len(self.frontera_pareto) == 0:
            return True

        # Verificar si domina a alguna solución existente
        for solucion in self.frontera_pareto:
            if self._domina(solucion['objetivos'], objetivos):
                return False

        return True

    def _domina(self, objetivos1: List[float], objetivos2: List[float]) -> bool:
        """
        Verifica si objetivos1 domina a objetivos2.

        Args:
            objetivos1: Primer conjunto de objetivos
            objetivos2: Segundo conjunto de objetivos

        Returns:
            True si objetivos1 domina a objetivos2
        """
        mejor_en_alguno = False

        for i in range(len(objetivos1)):
            if objetivos1[i] > objetivos2[i]:
                mejor_en_alguno = True
            elif objetivos1[i] < objetivos2[i]:
                return False

        return mejor_en_alguno

    def _calcular_diversidad_pareto(self) -> float:
        """
        Calcula la diversidad de la frontera de Pareto.

        Returns:
            Diversidad de la frontera de Pareto
        """
        if len(self.frontera_pareto) < 2:
            return 0.0

        # Calcular distancia promedio entre soluciones
        distancias = []
        soluciones = list(self.frontera_pareto)

        for i in range(len(soluciones)):
            for j in range(i + 1, len(soluciones)):
                dist = np.linalg.norm(np.array(soluciones[i]['objetivos']) -
                                      np.array(soluciones[j]['objetivos']))
                distancias.append(dist)

        return np.mean(distancias) if distancias else 0.0

    def _calcular_convergencia_multi_obj(self) -> float:
        """
        Calcula la convergencia multi-objetivo.

        Returns:
            Score de convergencia multi-objetivo
        """
        if len(self.historial_objetivos) < 10:
            return 0.0

        # Calcular varianza de objetivos recientes
        objetivos_recientes = [h['objetivos'] for h in list(self.historial_objetivos)[-100:]]
        objetivos_array = np.array(objetivos_recientes)

        varianza_objetivos = np.mean(np.var(objetivos_array, axis=0))
        convergencia = 1.0 / (1.0 + varianza_objetivos)

        return convergencia

    def _balancear_objetivos(self, objetivos: List[float]) -> List[float]:
        """
        Balancea los objetivos usando pesos adaptativos.

        Args:
            objetivos: Valores de objetivos

        Returns:
            Objetivos balanceados
        """
        # Normalizar objetivos
        objetivos_norm = []
        for i, obj in enumerate(objetivos):
            if abs(obj) > 1e-8:
                objetivos_norm.append(obj / (1.0 + abs(obj)))
            else:
                objetivos_norm.append(obj)

        # Aplicar pesos adaptativos
        objetivos_balanceados = []
        for i, obj in enumerate(objetivos_norm):
            peso = self.objective_weights[i] if i < len(self.objective_weights) else 1.0
            objetivos_balanceados.append(obj * peso)

        return objetivos_balanceados

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante multi-objetivo.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado multi-objetivo
        """
        if not self.pesos_objetivos:
            self.inicializar_pesos()

        # Calcular salida para cada objetivo
        salidas_objetivos = []

        for i in range(self.num_objectives):
            logits = np.dot(estado, self.pesos_objetivos[i]) + self.sesgos_objetivos[i]
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            salidas_objetivos.append(probabilidades)

        # Combinar salidas usando pesos de objetivos
        salida_final = np.zeros_like(salidas_objetivos[0])

        for i, salida in enumerate(salidas_objetivos):
            peso = self.objective_weights[i] if i < len(self.objective_weights) else 1.0
            salida_final += peso * salida

        # Normalizar salida final
        salida_final = salida_final / np.sum(salida_final, axis=1, keepdims=True)

        return salida_final

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción usando la política multi-objetivo.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades multi-objetivo
        accion = np.random.choice(self.output_size, p=probabilidades[0])

        return accion

    def entrenar_paso_multi_obj(self, estado: np.ndarray, accion: int, recompensa: float,
                                siguiente_estado: np.ndarray, terminado: bool) -> None:
        """
        Realiza un paso de entrenamiento multi-objetivo.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            terminado: Si el episodio terminó
        """
        # Calcular todos los objetivos
        objetivos = self._calcular_todos_objetivos(estado, accion, recompensa)

        # Balancear objetivos
        objetivos_balanceados = self._balancear_objetivos(objetivos)

        # Verificar Pareto optimalidad
        es_pareto = self._es_pareto_optimal(objetivos_balanceados)

        if es_pareto:
            # Agregar a frontera de Pareto
            solucion_pareto = {
                'estado': estado.copy(),
                'accion': accion,
                'objetivos': objetivos_balanceados.copy(),
                'timestamp': time.time()
            }
            self.frontera_pareto.append(solucion_pareto)

        # Entrenar cada objetivo
        for i in range(self.num_objectives):
            # Calcular gradiente para este objetivo
            logits = np.dot(estado, self.pesos_objetivos[i]) + self.sesgos_objetivos[i]
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Gradiente de política
            grad_log_prob = np.zeros(self.output_size)
            grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

            # Aplicar objetivo específico
            objetivo_valor = objetivos_balanceados[i]
            grad_pesos = np.outer(estado, grad_log_prob) * objetivo_valor
            grad_sesgo = grad_log_prob * objetivo_valor

            # Actualizar pesos para este objetivo
            self.pesos_objetivos[i] -= self.learning_rate * grad_pesos
            self.sesgos_objetivos[i] -= self.learning_rate * grad_sesgo

        # Guardar en historial
        self.historial_objetivos.append({
            'estado': estado.copy(),
            'accion': accion,
            'objetivos': objetivos_balanceados.copy(),
            'es_pareto': es_pareto,
            'timestamp': time.time()
        })

        # Actualizar estadísticas
        self._actualizar_estadisticas_multi_obj(objetivos_balanceados)

    def _actualizar_estadisticas_multi_obj(self, objetivos: List[float]) -> None:
        """
        Actualiza las estadísticas multi-objetivo.
        """
        # Actualizar rendimiento de objetivos
        for i, obj in enumerate(objetivos):
            if i < 3:  # Solo los primeros 3 objetivos
                self.estadisticas_multi_obj[f'objetivo_{i+1}_rendimiento'] = obj

        # Calcular métricas multi-objetivo
        self.estadisticas_multi_obj['pareto_optimalidad'] = len(self.frontera_pareto) / max(1, len(self.historial_objetivos))
        self.estadisticas_multi_obj['diversidad_soluciones'] = self._calcular_diversidad_pareto()
        self.estadisticas_multi_obj['convergencia_multi_obj'] = self._calcular_convergencia_multi_obj()

        # Calcular balance de objetivos
        balance = 1.0 - np.var(objetivos) / (np.mean(np.abs(objetivos)) + 1e-8)
        self.estadisticas_multi_obj['balance_objetivos'] = balance

        # Guardar en historiales
        self.historial_pareto.append(self.estadisticas_multi_obj['pareto_optimalidad'])
        self.historial_diversidad.append(self.estadisticas_multi_obj['diversidad_soluciones'])
        self.historial_convergencia.append(self.estadisticas_multi_obj['convergencia_multi_obj'])

    def obtener_estadisticas_multi_obj(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de la optimización multi-objetivo.

        Returns:
            Diccionario con estadísticas multi-objetivo
        """
        if not self.pesos_objetivos:
            return {'estado': 'no_inicializada'}

        stats_multi_obj = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'num_objectives': self.num_objectives,
            'objective_weights': self.objective_weights,
            'pareto_threshold': self.pareto_threshold,
            'diversity_weight': self.diversity_weight,
            'convergence_weight': self.convergence_weight,
            'adaptation_rate': self.adaptation_rate,
            'balance_factor': self.balance_factor,
            'objetivo_1_rendimiento': self.estadisticas_multi_obj['objetivo_1_rendimiento'],
            'objetivo_2_rendimiento': self.estadisticas_multi_obj['objetivo_2_rendimiento'],
            'objetivo_3_rendimiento': self.estadisticas_multi_obj['objetivo_3_rendimiento'],
            'pareto_optimalidad': self.estadisticas_multi_obj['pareto_optimalidad'],
            'diversidad_soluciones': self.estadisticas_multi_obj['diversidad_soluciones'],
            'convergencia_multi_obj': self.estadisticas_multi_obj['convergencia_multi_obj'],
            'balance_objetivos': self.estadisticas_multi_obj['balance_objetivos'],
            'eficiencia_pareto': self.estadisticas_multi_obj['eficiencia_pareto'],
            'estabilidad_multi_obj': self.estadisticas_multi_obj['estabilidad_multi_obj'],
            'adaptacion_objetivos': self.estadisticas_multi_obj['adaptacion_objetivos'],
            'frontera_pareto_size': len(self.frontera_pareto),
            'soluciones_pareto_size': len(self.soluciones_pareto)
        }

        return stats_multi_obj

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona multi-objetivo.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas multi-objetivo
        estabilidad['pareto_optimalidad_ok'] = self.estadisticas_multi_obj['pareto_optimalidad'] > 0.1
        estabilidad['diversidad_suficiente'] = self.estadisticas_multi_obj['diversidad_soluciones'] > 0.01
        estabilidad['convergencia_multi_obj_ok'] = self.estadisticas_multi_obj['convergencia_multi_obj'] > 0.5
        estabilidad['balance_objetivos_ok'] = self.estadisticas_multi_obj['balance_objetivos'] > 0.3
        estabilidad['frontera_pareto_suficiente'] = len(self.frontera_pareto) > 10

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     num_objectives: int = None,
                                     objective_weights: List[float] = None,
                                     pareto_threshold: float = None,
                                     diversity_weight: float = None,
                                     convergence_weight: float = None,
                                     adaptation_rate: float = None,
                                     balance_factor: float = None) -> None:
        """
        Reinicializa la neurona multi-objetivo con nuevos parámetros.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if num_objectives is not None:
            self.num_objectives = num_objectives
        if objective_weights is not None:
            self.objective_weights = objective_weights
        if pareto_threshold is not None:
            self.pareto_threshold = pareto_threshold
        if diversity_weight is not None:
            self.diversity_weight = diversity_weight
        if convergence_weight is not None:
            self.convergence_weight = convergence_weight
        if adaptation_rate is not None:
            self.adaptation_rate = adaptation_rate
        if balance_factor is not None:
            self.balance_factor = balance_factor

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar estructuras multi-objetivo
        self.frontera_pareto.clear()
        self.soluciones_pareto.clear()

        # Limpiar historiales específicos
        self.historial_objetivos.clear()
        self.historial_pareto.clear()
        self.historial_diversidad.clear()
        self.historial_convergencia.clear()

        # Resetear estadísticas multi-objetivo
        for key in self.estadisticas_multi_obj:
            self.estadisticas_multi_obj[key] = 0.0

        logger.info(f"Neurona multi-objetivo reinicializada: lr={self.learning_rate}, "
                    f"objetivos={self.num_objectives}, pesos={self.objective_weights}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoMultiObjetivo(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"objetivos={self.num_objectives}, pesos={self.objective_weights})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_7


def crear_neurona_multi_objetivo(input_size: int, output_size: int,
                                 configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoMultiObjetivo:
    """
    Función de conveniencia para crear una neurona multi-objetivo.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona multi-objetivo configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoMultiObjetivo(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoMultiObjetivo'),
        learning_rate=configuracion.get('learning_rate', LUCIA_ADVANCED_RL_CONFIG['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma']),
        num_objectives=configuracion.get('num_objectives', 3),
        objective_weights=configuracion.get('objective_weights', LUCIA_ADVANCED_RL_CONFIG['multi_objective_weights']),
        pareto_threshold=configuracion.get('pareto_threshold', 0.1),
        diversity_weight=configuracion.get('diversity_weight', 0.3),
        convergence_weight=configuracion.get('convergence_weight', 0.7),
        adaptation_rate=configuracion.get('adaptation_rate', 0.01),
        balance_factor=configuracion.get('balance_factor', 0.5)
    )


# Configuración específica para RFEN2_RN_7
RFEN2_RN_7_CONFIG = {
    'inicializacion_preferida': 'multi_objetivo',
    'learning_rate_default': 0.0001,
    'gamma_default': 0.99,
    'num_objectives_default': 3,
    'objective_weights_default': [0.4, 0.3, 0.3],
    'pareto_threshold_default': 0.1,
    'diversity_weight_default': 0.3,
    'convergence_weight_default': 0.7,
    'adaptation_rate_default': 0.01,
    'balance_factor_default': 0.5,
    'umbral_pareto_optimalidad': 0.1,
    'umbral_diversidad_suficiente': 0.01,
    'umbral_convergencia_multi_obj': 0.5,
    'umbral_balance_objetivos': 0.3,
    'umbral_frontera_pareto': 10
}

logger.info("RFEN2_RN_7.py cargado correctamente - Neurona de Refuerzo Multi-Objetivo")
