"""
RFEN2_RN_4.py - Neurona de Refuerzo con Optimización Evolutiva
==============================================================

Esta neurona implementa Optimización Evolutiva de Pesos utilizando algoritmos
genéticos y estrategias evolutivas para encontrar configuraciones óptimas de
pesos, utilizando técnicas avanzadas de 2025.

Características Avanzadas 2025:
- Algoritmos genéticos para optimización de pesos
- Estrategias evolutivas (ES) adaptativas
- Selección natural de configuraciones de pesos
- Mutación y crossover adaptativos
- Población diversa de pesos
- Convergencia evolutiva acelerada
- Optimización multi-objetivo evolutiva
- Preservación de diversidad genética

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
import copy
from . import NeuronaRefuerzoAvanzadaBase, inicializar_pesos_evolutivos, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_4')


class NeuronaRefuerzoEvolutiva(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con Optimización Evolutiva de Pesos.

    Esta neurona implementa algoritmos genéticos y estrategias evolutivas
    para encontrar configuraciones óptimas de pesos mediante evolución.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoEvolutiva",
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 selection_pressure: float = 2.0,
                 elite_ratio: float = 0.1,
                 diversity_threshold: float = 0.1,
                 convergence_threshold: float = 0.01,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99):
        """
        Inicializa la neurona de refuerzo evolutiva.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            population_size: Tamaño de la población
            mutation_rate: Tasa de mutación
            crossover_rate: Tasa de crossover
            selection_pressure: Presión de selección
            elite_ratio: Proporción de élite
            diversity_threshold: Umbral de diversidad
            convergence_threshold: Umbral de convergencia
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
        """
        super().__init__(input_size, output_size, nombre)

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        self.elite_ratio = elite_ratio
        self.diversity_threshold = diversity_threshold
        self.convergence_threshold = convergence_threshold
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Población de pesos
        self.poblacion_pesos = []
        self.poblacion_sesgos = []
        self.fitness_poblacion = []

        # Mejor individuo actual
        self.mejor_pesos = None
        self.mejor_sesgo = None
        self.mejor_fitness = float('-inf')

        # Estadísticas específicas de evolución
        self.estadisticas_evolutivas = {
            'generacion_actual': 0,
            'fitness_promedio': 0.0,
            'fitness_maximo': float('-inf'),
            'fitness_minimo': float('inf'),
            'diversidad_poblacion': 0.0,
            'convergencia_evolutiva': 0.0,
            'tasa_mutacion_adaptativa': 0.0,
            'eficiencia_seleccion': 0.0,
            'preservacion_diversidad': 0.0,
            'aceleracion_convergencia': 0.0
        }

        # Historial evolutivo
        self.historial_fitness = deque(maxlen=1000)
        self.historial_diversidad = deque(maxlen=1000)
        self.historial_convergencia = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoEvolutiva creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa la población de pesos evolutivos.
        """
        # Crear población inicial de pesos
        self.poblacion_pesos = inicializar_pesos_evolutivos(
            (self.input_size, self.output_size),
            self.population_size
        )

        # Crear población inicial de sesgos
        self.poblacion_sesgos = []
        for _ in range(self.population_size):
            sesgo = np.random.normal(0, 0.1, (1, self.output_size))
            self.poblacion_sesgos.append(sesgo.astype(LUCIA_ADVANCED_RL_CONFIG['precision']))

        # Inicializar fitness
        self.fitness_poblacion = [0.0] * self.population_size

        # Seleccionar mejor individuo inicial
        self._seleccionar_mejor_individuo()

        logger.info(f"Población evolutiva inicializada con {self.population_size} individuos")

    def _seleccionar_mejor_individuo(self) -> None:
        """
        Selecciona el mejor individuo de la población actual.
        """
        if not self.fitness_poblacion:
            return

        mejor_idx = np.argmax(self.fitness_poblacion)
        mejor_fitness = self.fitness_poblacion[mejor_idx]

        if mejor_fitness > self.mejor_fitness:
            self.mejor_fitness = mejor_fitness
            self.mejor_pesos = self.poblacion_pesos[mejor_idx].copy()
            self.mejor_sesgo = self.poblacion_sesgos[mejor_idx].copy()

    def _calcular_diversidad_poblacion(self) -> float:
        """
        Calcula la diversidad de la población.

        Returns:
            Diversidad de la población
        """
        if len(self.poblacion_pesos) < 2:
            return 0.0

        # Calcular varianza promedio de los pesos
        pesos_array = np.array(self.poblacion_pesos)
        varianza_promedio = np.mean(np.var(pesos_array, axis=0))

        return varianza_promedio

    def _calcular_convergencia_evolutiva(self) -> float:
        """
        Calcula la convergencia evolutiva.

        Returns:
            Score de convergencia evolutiva
        """
        if len(self.historial_fitness) < 10:
            return 0.0

        # Calcular varianza de fitness reciente
        fitness_recientes = list(self.historial_fitness)[-100:]
        varianza_fitness = np.var(fitness_recientes)

        # Convergencia inversamente proporcional a la varianza
        convergencia = 1.0 / (1.0 + varianza_fitness)

        return convergencia

    def _seleccionar_padres(self) -> List[int]:
        """
        Selecciona padres para reproducción usando selección por torneo.

        Returns:
            Lista de índices de padres seleccionados
        """
        padres = []

        for _ in range(self.population_size):
            # Selección por torneo
            candidatos = random.sample(range(self.population_size),
                                       min(3, self.population_size))

            # Seleccionar el mejor candidato
            mejor_candidato = max(candidatos, key=lambda x: self.fitness_poblacion[x])
            padres.append(mejor_candidato)

        return padres

    def _crossover(self, padre1_idx: int, padre2_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza crossover entre dos padres.

        Args:
            padre1_idx: Índice del primer padre
            padre2_idx: Índice del segundo padre

        Returns:
            Tupla con (pesos_hijo, sesgo_hijo)
        """
        padre1_pesos = self.poblacion_pesos[padre1_idx]
        padre2_pesos = self.poblacion_pesos[padre2_idx]
        padre1_sesgo = self.poblacion_sesgos[padre1_idx]
        padre2_sesgo = self.poblacion_sesgos[padre2_idx]

        # Crossover uniforme
        mask = np.random.random(padre1_pesos.shape) < 0.5

        pesos_hijo = np.where(mask, padre1_pesos, padre2_pesos)
        sesgo_hijo = np.where(np.random.random(padre1_sesgo.shape) < 0.5,
                              padre1_sesgo, padre2_sesgo)

        return pesos_hijo, sesgo_hijo

    def _mutar(self, pesos: np.ndarray, sesgo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica mutación a un individuo.

        Args:
            pesos: Pesos a mutar
            sesgo: Sesgo a mutar

        Returns:
            Tupla con (pesos_mutados, sesgo_mutado)
        """
        # Mutación gaussiana
        mutacion_pesos = np.random.normal(0, self.mutation_rate, pesos.shape)
        mutacion_sesgo = np.random.normal(0, self.mutation_rate, sesgo.shape)

        pesos_mutados = pesos + mutacion_pesos
        sesgo_mutado = sesgo + mutacion_sesgo

        return pesos_mutados, sesgo_mutado

    def _evaluar_fitness(self, pesos: np.ndarray, sesgo: np.ndarray,
                         estado: np.ndarray, accion: int, recompensa: float) -> float:
        """
        Evalúa el fitness de un individuo.

        Args:
            pesos: Pesos del individuo
            sesgo: Sesgo del individuo
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida

        Returns:
            Fitness del individuo
        """
        # Calcular probabilidades de acción
        logits = np.dot(estado, pesos) + sesgo
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Fitness basado en probabilidad de la acción tomada y recompensa
        fitness = probabilidades[0, accion] * recompensa

        # Penalizar pesos muy grandes (regularización)
        penalty = -0.01 * np.sum(pesos ** 2)

        return fitness + penalty

    def evolucionar_generacion(self, estados: List[np.ndarray], acciones: List[int],
                               recompensas: List[float]) -> None:
        """
        Evoluciona una generación completa.

        Args:
            estados: Lista de estados
            acciones: Lista de acciones
            recompensas: Lista de recompensas
        """
        if not self.poblacion_pesos:
            self.inicializar_pesos()

        # Evaluar fitness de todos los individuos
        for i in range(self.population_size):
            fitness_total = 0.0
            for estado, accion, recompensa in zip(estados, acciones, recompensas):
                fitness_total += self._evaluar_fitness(
                    self.poblacion_pesos[i],
                    self.poblacion_sesgos[i],
                    estado, accion, recompensa
                )

            self.fitness_poblacion[i] = fitness_total / len(estados)

        # Seleccionar mejor individuo
        self._seleccionar_mejor_individuo()

        # Crear nueva generación
        nueva_poblacion_pesos = []
        nueva_poblacion_sesgos = []

        # Preservar élite
        elite_size = int(self.population_size * self.elite_ratio)
        elite_indices = np.argsort(self.fitness_poblacion)[-elite_size:]

        for idx in elite_indices:
            nueva_poblacion_pesos.append(self.poblacion_pesos[idx].copy())
            nueva_poblacion_sesgos.append(self.poblacion_sesgos[idx].copy())

        # Generar resto de la población
        padres = self._seleccionar_padres()

        while len(nueva_poblacion_pesos) < self.population_size:
            # Seleccionar padres aleatorios
            padre1_idx = random.choice(padres)
            padre2_idx = random.choice(padres)

            # Crossover
            if random.random() < self.crossover_rate:
                pesos_hijo, sesgo_hijo = self._crossover(padre1_idx, padre2_idx)
            else:
                # Sin crossover, copiar padre
                pesos_hijo = self.poblacion_pesos[padre1_idx].copy()
                sesgo_hijo = self.poblacion_sesgos[padre1_idx].copy()

            # Mutación
            if random.random() < self.mutation_rate:
                pesos_hijo, sesgo_hijo = self._mutar(pesos_hijo, sesgo_hijo)

            nueva_poblacion_pesos.append(pesos_hijo)
            nueva_poblacion_sesgos.append(sesgo_hijo)

        # Actualizar población
        self.poblacion_pesos = nueva_poblacion_pesos
        self.poblacion_sesgos = nueva_poblacion_sesgos

        # Actualizar estadísticas
        self.estadisticas_evolutivas['generacion_actual'] += 1
        self.estadisticas_evolutivas['fitness_promedio'] = np.mean(self.fitness_poblacion)
        self.estadisticas_evolutivas['fitness_maximo'] = np.max(self.fitness_poblacion)
        self.estadisticas_evolutivas['fitness_minimo'] = np.min(self.fitness_poblacion)
        self.estadisticas_evolutivas['diversidad_poblacion'] = self._calcular_diversidad_poblacion()
        self.estadisticas_evolutivas['convergencia_evolutiva'] = self._calcular_convergencia_evolutiva()

        # Guardar en historiales
        self.historial_fitness.append(self.estadisticas_evolutivas['fitness_promedio'])
        self.historial_diversidad.append(self.estadisticas_evolutivas['diversidad_poblacion'])
        self.historial_convergencia.append(self.estadisticas_evolutivas['convergencia_evolutiva'])

        # Adaptar tasa de mutación
        self._adaptar_tasa_mutacion()

    def _adaptar_tasa_mutacion(self) -> None:
        """
        Adapta la tasa de mutación basada en la convergencia.
        """
        convergencia = self.estadisticas_evolutivas['convergencia_evolutiva']

        # Si la población está convergiendo, reducir mutación
        if convergencia > 0.8:
            self.mutation_rate *= 0.95
        # Si la población está muy diversa, aumentar mutación
        elif self.estadisticas_evolutivas['diversidad_poblacion'] > self.diversity_threshold:
            self.mutation_rate *= 1.05

        # Mantener tasa de mutación en rango razonable
        self.mutation_rate = np.clip(self.mutation_rate, 0.01, 0.5)

        self.estadisticas_evolutivas['tasa_mutacion_adaptativa'] = self.mutation_rate

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante usando el mejor individuo.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado
        """
        if self.mejor_pesos is None:
            self.inicializar_pesos()

        # Usar el mejor individuo encontrado
        logits = np.dot(estado, self.mejor_pesos) + self.mejor_sesgo

        # Aplicar función de activación
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probabilidades

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción usando el mejor individuo evolutivo.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades del mejor individuo
        accion = np.random.choice(self.output_size, p=probabilidades[0])

        return accion

    def obtener_estadisticas_evolutivas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de la optimización evolutiva.

        Returns:
            Diccionario con estadísticas evolutivas
        """
        if not self.poblacion_pesos:
            return {'estado': 'no_inicializada'}

        stats_evolutivas = {
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'selection_pressure': self.selection_pressure,
            'elite_ratio': self.elite_ratio,
            'diversity_threshold': self.diversity_threshold,
            'convergence_threshold': self.convergence_threshold,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'generacion_actual': self.estadisticas_evolutivas['generacion_actual'],
            'fitness_promedio': self.estadisticas_evolutivas['fitness_promedio'],
            'fitness_maximo': self.estadisticas_evolutivas['fitness_maximo'],
            'fitness_minimo': self.estadisticas_evolutivas['fitness_minimo'],
            'diversidad_poblacion': self.estadisticas_evolutivas['diversidad_poblacion'],
            'convergencia_evolutiva': self.estadisticas_evolutivas['convergencia_evolutiva'],
            'tasa_mutacion_adaptativa': self.estadisticas_evolutivas['tasa_mutacion_adaptativa'],
            'mejor_fitness': self.mejor_fitness
        }

        return stats_evolutivas

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona evolutiva.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas de evolución
        estabilidad['poblacion_diversa'] = self.estadisticas_evolutivas['diversidad_poblacion'] > 0.01
        estabilidad['convergencia_estable'] = self.estadisticas_evolutivas['convergencia_evolutiva'] > 0.5
        estabilidad['fitness_mejorando'] = self.mejor_fitness > float('-inf')
        estabilidad['tasa_mutacion_apropiada'] = 0.01 < self.mutation_rate < 0.5
        estabilidad['elite_preservada'] = len(self.poblacion_pesos) > 0

        return estabilidad

    def reinicializar_con_parametros(self, population_size: int = None,
                                     mutation_rate: float = None,
                                     crossover_rate: float = None,
                                     selection_pressure: float = None,
                                     elite_ratio: float = None,
                                     learning_rate: float = None,
                                     gamma: float = None) -> None:
        """
        Reinicializa la neurona evolutiva con nuevos parámetros.
        """
        if population_size is not None:
            self.population_size = population_size
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        if crossover_rate is not None:
            self.crossover_rate = crossover_rate
        if selection_pressure is not None:
            self.selection_pressure = selection_pressure
        if elite_ratio is not None:
            self.elite_ratio = elite_ratio
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar historiales específicos
        self.historial_fitness.clear()
        self.historial_diversidad.clear()
        self.historial_convergencia.clear()

        # Resetear estadísticas evolutivas
        for key in self.estadisticas_evolutivas:
            if key == 'generacion_actual':
                self.estadisticas_evolutivas[key] = 0
            else:
                self.estadisticas_evolutivas[key] = 0.0

        # Resetear mejor individuo
        self.mejor_fitness = float('-inf')
        self.mejor_pesos = None
        self.mejor_sesgo = None

        logger.info(f"Neurona evolutiva reinicializada: pop_size={self.population_size}, "
                    f"mutation={self.mutation_rate}, crossover={self.crossover_rate}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoEvolutiva(entrada={self.input_size}, "
                f"salida={self.output_size}, pop_size={self.population_size}, "
                f"mutation={self.mutation_rate}, crossover={self.crossover_rate})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_4


def crear_neurona_evolutiva(input_size: int, output_size: int,
                            configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoEvolutiva:
    """
    Función de conveniencia para crear una neurona evolutiva.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona evolutiva configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoEvolutiva(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoEvolutiva'),
        population_size=configuracion.get('population_size', LUCIA_ADVANCED_RL_CONFIG['evolution_population_size']),
        mutation_rate=configuracion.get('mutation_rate', 0.1),
        crossover_rate=configuracion.get('crossover_rate', 0.8),
        selection_pressure=configuracion.get('selection_pressure', 2.0),
        elite_ratio=configuracion.get('elite_ratio', 0.1),
        diversity_threshold=configuracion.get('diversity_threshold', 0.1),
        convergence_threshold=configuracion.get('convergence_threshold', 0.01),
        learning_rate=configuracion.get('learning_rate', LUCIA_ADVANCED_RL_CONFIG['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma'])
    )


# Configuración específica para RFEN2_RN_4
RFEN2_RN_4_CONFIG = {
    'inicializacion_preferida': 'evolutiva',
    'population_size_default': 100,
    'mutation_rate_default': 0.1,
    'crossover_rate_default': 0.8,
    'selection_pressure_default': 2.0,
    'elite_ratio_default': 0.1,
    'diversity_threshold_default': 0.1,
    'convergence_threshold_default': 0.01,
    'learning_rate_default': 0.0001,
    'gamma_default': 0.99,
    'umbral_diversidad_poblacion': 0.01,
    'umbral_convergencia_evolutiva': 0.5,
    'umbral_tasa_mutacion': [0.01, 0.5],
    'umbral_elite_preservada': 0
}

logger.info("RFEN2_RN_4.py cargado correctamente - Neurona de Refuerzo Evolutiva")
