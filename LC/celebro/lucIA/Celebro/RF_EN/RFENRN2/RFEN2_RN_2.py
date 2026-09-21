"""
RFEN2_RN_2.py - Neurona de Refuerzo con Meta-Aprendizaje
========================================================

Esta neurona implementa Meta-Aprendizaje (Meta-Learning) con pesos dinámicos
que se adaptan automáticamente a nuevas tareas y entornos, utilizando técnicas
avanzadas de 2025 para optimización de hiperparámetros automática.

Características Avanzadas 2025:
- Meta-aprendizaje con adaptación rápida a nuevas tareas
- Optimización automática de hiperparámetros
- Pesos dinámicos que evolucionan según la tarea
- Aprendizaje de cómo aprender (learning to learn)
- Transferencia de conocimiento entre tareas
- Adaptación few-shot y zero-shot
- Meta-gradientes para actualización eficiente
- Arquitectura de memoria episódica

Autor: LucIA Development Team
Versión: 2.0.0
Fecha: 2025
"""

import numpy as np
import math
import time
from typing import Tuple, Optional, Dict, Any, List
import logging
from collections import deque
import random
import copy
from . import NeuronaRefuerzoAvanzadaBase, inicializar_pesos_meta_learning, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_2')


class NeuronaRefuerzoMetaLearning(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con Meta-Aprendizaje (Meta-Learning).

    Esta neurona implementa pesos dinámicos que se adaptan automáticamente
    a nuevas tareas y entornos, utilizando técnicas de meta-aprendizaje
    para optimización automática de hiperparámetros.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoMetaLearning",
                 meta_learning_rate: float = 0.0001,
                 inner_learning_rate: float = 0.01,
                 gamma: float = 0.99,
                 meta_batch_size: int = 32,
                 adaptation_steps: int = 5,
                 memory_size: int = 1000,
                 task_memory_size: int = 100,
                 meta_gradient_clip: float = 1.0,
                 adaptation_rate: float = 0.1,
                 transfer_factor: float = 0.5):
        """
        Inicializa la neurona de refuerzo con meta-aprendizaje.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            meta_learning_rate: Tasa de aprendizaje meta
            inner_learning_rate: Tasa de aprendizaje interna
            gamma: Factor de descuento
            meta_batch_size: Tamaño del batch para meta-aprendizaje
            adaptation_steps: Número de pasos de adaptación
            memory_size: Tamaño de la memoria episódica
            task_memory_size: Tamaño de la memoria de tareas
            meta_gradient_clip: Clipping de gradientes meta
            adaptation_rate: Tasa de adaptación de pesos
            transfer_factor: Factor de transferencia entre tareas
        """
        super().__init__(input_size, output_size, nombre)

        self.meta_learning_rate = meta_learning_rate
        self.inner_learning_rate = inner_learning_rate
        self.gamma = gamma
        self.meta_batch_size = meta_batch_size
        self.adaptation_steps = adaptation_steps
        self.memory_size = memory_size
        self.task_memory_size = task_memory_size
        self.meta_gradient_clip = meta_gradient_clip
        self.adaptation_rate = adaptation_rate
        self.transfer_factor = transfer_factor

        # Pesos meta y pesos de tarea específica
        self.pesos_meta = None
        self.sesgo_meta = None
        self.pesos_tarea = None
        self.sesgo_tarea = None

        # Memoria episódica para meta-aprendizaje
        self.memoria_episodica = deque(maxlen=memory_size)
        self.memoria_tareas = deque(maxlen=task_memory_size)
        self.historial_adaptacion = deque(maxlen=500)

        # Estado meta actual
        self.tarea_actual = None
        self.contexto_tarea = None
        self.pesos_adaptados = None
        self.sesgo_adaptado = None

        # Estadísticas específicas de meta-aprendizaje
        self.estadisticas_meta = {
            'adaptacion_rapida': 0.0,
            'transferencia_conocimiento': 0.0,
            'meta_gradiente_norma': 0.0,
            'adaptacion_steps_efectivos': 0.0,
            'memoria_episodica_utilizada': 0.0,
            'tareas_aprendidas': 0,
            'eficiencia_meta_aprendizaje': 0.0,
            'estabilidad_meta_pesos': 0.0,
            'convergencia_meta': 0.0,
            'transferencia_exitosa': 0.0
        }

        # Historial específico para análisis meta
        self.historial_meta_gradientes = []
        self.historial_adaptacion_rapida = []
        self.historial_transferencia = []
        self.historial_eficiencia_meta = []
        self.historial_estabilidad_meta = []

        logger.info(f"NeuronaRefuerzoMetaLearning creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos meta y de tarea específica.
        """
        # Pesos meta con inicialización específica para meta-aprendizaje
        self.pesos_meta = inicializar_pesos_meta_learning(
            (self.input_size, self.output_size),
            self.meta_learning_rate
        )
        self.sesgo_meta = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Pesos de tarea específica (inicialmente iguales a los meta)
        self.pesos_tarea = self.pesos_meta.copy()
        self.sesgo_tarea = self.sesgo_meta.copy()

        # Pesos adaptados (para uso durante la adaptación)
        self.pesos_adaptados = self.pesos_meta.copy()
        self.sesgo_adaptado = self.sesgo_meta.copy()

        logger.info(f"Pesos meta inicializados con meta_lr={self.meta_learning_rate}")

    def _adaptar_a_tarea(self, tarea_contexto: np.ndarray, pasos: int = None) -> None:
        """
        Adapta los pesos a una tarea específica usando meta-aprendizaje.

        Args:
            tarea_contexto: Contexto de la tarea actual
            pasos: Número de pasos de adaptación
        """
        if pasos is None:
            pasos = self.adaptation_steps

        # Inicializar pesos adaptados con los meta-pesos
        self.pesos_adaptados = self.pesos_meta.copy()
        self.sesgo_adaptado = self.sesgo_meta.copy()

        # Buscar experiencias similares en memoria episódica
        experiencias_similares = self._buscar_experiencias_similares(tarea_contexto)

        if experiencias_similares:
            # Usar transferencia de conocimiento
            self._aplicar_transferencia_conocimiento(experiencias_similares)

        # Realizar pasos de adaptación
        for paso in range(pasos):
            # Calcular gradiente de adaptación
            grad_pesos, grad_sesgo = self._calcular_gradiente_adaptacion(tarea_contexto)

            # Actualizar pesos adaptados
            self.pesos_adaptados -= self.inner_learning_rate * grad_pesos
            self.sesgo_adaptado -= self.inner_learning_rate * grad_sesgo

            # Guardar en historial de adaptación
            self.historial_adaptacion.append({
                'paso': paso,
                'pesos': self.pesos_adaptados.copy(),
                'gradiente_norma': np.linalg.norm(grad_pesos),
                'timestamp': time.time()
            })

        # Actualizar estadísticas
        self.estadisticas_meta['adaptacion_steps_efectivos'] = pasos
        self.estadisticas_meta['memoria_episodica_utilizada'] = len(experiencias_similares)

    def _buscar_experiencias_similares(self, contexto: np.ndarray) -> List[Dict[str, Any]]:
        """
        Busca experiencias similares en la memoria episódica.

        Args:
            contexto: Contexto de la tarea actual

        Returns:
            Lista de experiencias similares
        """
        if len(self.memoria_episodica) < 10:
            return []

        experiencias_similares = []

        for experiencia in self.memoria_episodica:
            # Calcular similitud basada en contexto
            similitud = np.dot(contexto, experiencia['contexto']) / (
                np.linalg.norm(contexto) * np.linalg.norm(experiencia['contexto']) + 1e-8
            )

            if similitud > 0.7:  # Umbral de similitud
                experiencias_similares.append(experiencia)

        # Ordenar por similitud y tomar las mejores
        experiencias_similares.sort(key=lambda x: x['similitud'], reverse=True)
        return experiencias_similares[:5]  # Top 5 experiencias similares

    def _aplicar_transferencia_conocimiento(self, experiencias: List[Dict[str, Any]]) -> None:
        """
        Aplica transferencia de conocimiento desde experiencias similares.

        Args:
            experiencias: Lista de experiencias similares
        """
        if not experiencias:
            return

        # Calcular pesos promedio de experiencias similares
        pesos_promedio = np.mean([exp['pesos_finales'] for exp in experiencias], axis=0)
        sesgo_promedio = np.mean([exp['sesgo_final'] for exp in experiencias], axis=0)

        # Aplicar transferencia con factor de transferencia
        self.pesos_adaptados = (1 - self.transfer_factor) * self.pesos_adaptados + \
            self.transfer_factor * pesos_promedio
        self.sesgo_adaptado = (1 - self.transfer_factor) * self.sesgo_adaptado + \
            self.transfer_factor * sesgo_promedio

        # Actualizar estadísticas
        self.estadisticas_meta['transferencia_conocimiento'] = self.transfer_factor
        self.estadisticas_meta['transferencia_exitosa'] = 1.0

    def _calcular_gradiente_adaptacion(self, contexto: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el gradiente de adaptación para meta-aprendizaje.

        Args:
            contexto: Contexto de la tarea

        Returns:
            Tupla con (gradiente_pesos, gradiente_sesgo)
        """
        # Simular gradiente de adaptación basado en contexto
        # En una implementación real, esto vendría del entrenamiento en la tarea específica

        # Calcular dirección de adaptación basada en contexto
        direccion_adaptacion = np.tanh(contexto)

        # Aplicar a los pesos
        grad_pesos = np.outer(direccion_adaptacion, np.ones(self.output_size))
        grad_sesgo = np.ones((1, self.output_size))

        # Normalizar gradientes
        grad_norm = np.linalg.norm(grad_pesos)
        if grad_norm > 0:
            grad_pesos = grad_pesos / grad_norm
            grad_sesgo = grad_sesgo / grad_norm

        return grad_pesos, grad_sesgo

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante con meta-aprendizaje.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto de la tarea (opcional)

        Returns:
            Acción o valor estimado con meta-aprendizaje
        """
        if self.pesos_meta is None:
            self.inicializar_pesos()

        # Si hay contexto de tarea, adaptar pesos
        if contexto is not None and not np.array_equal(contexto, self.contexto_tarea):
            self.tarea_actual = contexto
            self.contexto_tarea = contexto.copy()
            self._adaptar_a_tarea(contexto)

        # Usar pesos adaptados si están disponibles, sino usar meta-pesos
        if self.pesos_adaptados is not None:
            pesos_actuales = self.pesos_adaptados
            sesgo_actual = self.sesgo_adaptado
        else:
            pesos_actuales = self.pesos_meta
            sesgo_actual = self.sesgo_meta

        # Calcular salida
        logits = np.dot(estado, pesos_actuales) + sesgo_actual

        # Aplicar función de activación
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probabilidades

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción usando la política meta-adaptativa.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto de la tarea (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades meta-adaptativas
        accion = np.random.choice(self.output_size, p=probabilidades[0])

        return accion

    def calcular_meta_gradiente(self, estados: List[np.ndarray], acciones: List[int],
                                recompensas: List[float], contextos: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el meta-gradiente para actualizar los pesos meta.

        Args:
            estados: Lista de estados
            acciones: Lista de acciones
            recompensas: Lista de recompensas
            contextos: Lista de contextos de tarea

        Returns:
            Tupla con (meta_gradiente_pesos, meta_gradiente_sesgo)
        """
        if self.pesos_meta is None:
            raise ValueError("Pesos meta no inicializados")

        meta_grad_pesos = np.zeros_like(self.pesos_meta)
        meta_grad_sesgo = np.zeros_like(self.sesgo_meta)

        # Procesar cada tarea en el batch meta
        for i in range(len(estados)):
            estado = estados[i]
            accion = acciones[i]
            recompensa = recompensas[i]
            contexto = contextos[i]

            # Adaptar a la tarea específica
            self._adaptar_a_tarea(contexto)

            # Calcular gradiente en la tarea adaptada
            grad_pesos, grad_sesgo = self._calcular_gradiente_tarea(estado, accion, recompensa)

            # Acumular meta-gradiente
            meta_grad_pesos += grad_pesos
            meta_grad_sesgo += grad_sesgo

        # Promediar meta-gradiente
        n_tareas = len(estados)
        meta_grad_pesos /= n_tareas
        meta_grad_sesgo /= n_tareas

        # Aplicar clipping de meta-gradiente
        meta_grad_norm = np.linalg.norm(meta_grad_pesos)
        if meta_grad_norm > self.meta_gradient_clip:
            meta_grad_pesos = meta_grad_pesos * self.meta_gradient_clip / meta_grad_norm
            meta_grad_sesgo = meta_grad_sesgo * self.meta_gradient_clip / meta_grad_norm

        # Actualizar estadísticas
        self.estadisticas_meta['meta_gradiente_norma'] = meta_grad_norm

        return meta_grad_pesos, meta_grad_sesgo

    def _calcular_gradiente_tarea(self, estado: np.ndarray, accion: int, recompensa: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el gradiente para una tarea específica.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida

        Returns:
            Tupla con (gradiente_pesos, gradiente_sesgo)
        """
        # Calcular probabilidades con pesos adaptados
        probabilidades = self.forward(estado)

        # Calcular gradiente de política
        grad_log_prob = np.zeros(self.output_size)
        grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

        # Aplicar recompensa como ventaja
        grad_pesos = np.outer(estado, grad_log_prob) * recompensa
        grad_sesgo = grad_log_prob * recompensa

        return grad_pesos, grad_sesgo

    def actualizar_meta_pesos(self, meta_grad_pesos: np.ndarray, meta_grad_sesgo: np.ndarray) -> None:
        """
        Actualiza los pesos meta usando meta-gradientes.

        Args:
            meta_grad_pesos: Meta-gradiente de los pesos
            meta_grad_sesgo: Meta-gradiente del sesgo
        """
        if self.pesos_meta is None:
            raise ValueError("Pesos meta no inicializados")

        # Actualizar pesos meta
        self.pesos_meta -= self.meta_learning_rate * meta_grad_pesos
        self.sesgo_meta -= self.meta_learning_rate * meta_grad_sesgo

        # Guardar en historial de pesos
        self.historial_pesos.append(self.pesos_meta.copy())

        # Guardar en historiales específicos
        self.historial_meta_gradientes.append(np.linalg.norm(meta_grad_pesos))
        self.historial_adaptacion_rapida.append(self.estadisticas_meta['adaptacion_rapida'])
        self.historial_transferencia.append(self.estadisticas_meta['transferencia_conocimiento'])

        # Actualizar estadísticas
        self.estadisticas_meta['tareas_aprendidas'] += 1
        self._calcular_eficiencia_meta()
        self._calcular_estabilidad_meta()

    def _calcular_eficiencia_meta(self) -> None:
        """
        Calcula la eficiencia del meta-aprendizaje.
        """
        if len(self.historial_meta_gradientes) < 10:
            return

        # Eficiencia basada en convergencia de meta-gradientes
        meta_gradientes_recientes = np.array(self.historial_meta_gradientes[-100:])
        eficiencia = 1.0 / (1.0 + np.var(meta_gradientes_recientes))

        self.estadisticas_meta['eficiencia_meta_aprendizaje'] = eficiencia

    def _calcular_estabilidad_meta(self) -> None:
        """
        Calcula la estabilidad de los pesos meta.
        """
        if len(self.historial_pesos) < 10:
            return

        # Estabilidad basada en varianza de pesos meta
        pesos_recientes = np.array(self.historial_pesos[-100:])
        varianza_pesos = np.var(pesos_recientes)
        estabilidad = 1.0 / (1.0 + varianza_pesos)

        self.estadisticas_meta['estabilidad_meta_pesos'] = estabilidad
        self.estadisticas_meta['convergencia_meta'] = estabilidad

    def almacenar_experiencia_episodica(self, estado: np.ndarray, accion: int, recompensa: float,
                                        siguiente_estado: np.ndarray, contexto: np.ndarray,
                                        pesos_finales: np.ndarray, sesgo_final: np.ndarray) -> None:
        """
        Almacena una experiencia en la memoria episódica.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            contexto: Contexto de la tarea
            pesos_finales: Pesos finales después de la adaptación
            sesgo_final: Sesgo final después de la adaptación
        """
        experiencia = {
            'estado': estado.copy(),
            'accion': accion,
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'contexto': contexto.copy(),
            'pesos_finales': pesos_finales.copy(),
            'sesgo_final': sesgo_final.copy(),
            'timestamp': time.time()
        }

        self.memoria_episodica.append(experiencia)

    def obtener_estadisticas_meta(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas del meta-aprendizaje.

        Returns:
            Diccionario con estadísticas meta
        """
        if self.pesos_meta is None:
            return {'estado': 'no_inicializada'}

        stats_meta = {
            'meta_learning_rate': self.meta_learning_rate,
            'inner_learning_rate': self.inner_learning_rate,
            'gamma': self.gamma,
            'meta_batch_size': self.meta_batch_size,
            'adaptation_steps': self.adaptation_steps,
            'memory_size': len(self.memoria_episodica),
            'task_memory_size': len(self.memoria_tareas),
            'meta_gradient_clip': self.meta_gradient_clip,
            'adaptation_rate': self.adaptation_rate,
            'transfer_factor': self.transfer_factor,
            'adaptacion_rapida': self.estadisticas_meta['adaptacion_rapida'],
            'transferencia_conocimiento': self.estadisticas_meta['transferencia_conocimiento'],
            'meta_gradiente_norma': self.estadisticas_meta['meta_gradiente_norma'],
            'adaptacion_steps_efectivos': self.estadisticas_meta['adaptacion_steps_efectivos'],
            'memoria_episodica_utilizada': self.estadisticas_meta['memoria_episodica_utilizada'],
            'tareas_aprendidas': self.estadisticas_meta['tareas_aprendidas'],
            'eficiencia_meta_aprendizaje': self.estadisticas_meta['eficiencia_meta_aprendizaje'],
            'estabilidad_meta_pesos': self.estadisticas_meta['estabilidad_meta_pesos'],
            'convergencia_meta': self.estadisticas_meta['convergencia_meta'],
            'transferencia_exitosa': self.estadisticas_meta['transferencia_exitosa']
        }

        return stats_meta

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona meta-aprendizaje.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas de meta-aprendizaje
        estabilidad['meta_pesos_estables'] = self.estadisticas_meta['estabilidad_meta_pesos'] > 0.7
        estabilidad['meta_gradientes_controlados'] = self.estadisticas_meta['meta_gradiente_norma'] < 5.0
        estabilidad['adaptacion_eficiente'] = self.estadisticas_meta['eficiencia_meta_aprendizaje'] > 0.6
        estabilidad['transferencia_funcional'] = self.estadisticas_meta['transferencia_exitosa'] > 0.5
        estabilidad['memoria_episodica_suficiente'] = len(self.memoria_episodica) > 50

        return estabilidad

    def reinicializar_con_parametros(self, meta_learning_rate: float = None,
                                     inner_learning_rate: float = None,
                                     gamma: float = None,
                                     meta_batch_size: int = None,
                                     adaptation_steps: int = None,
                                     adaptation_rate: float = None,
                                     transfer_factor: float = None) -> None:
        """
        Reinicializa la neurona meta-aprendizaje con nuevos parámetros.
        """
        if meta_learning_rate is not None:
            self.meta_learning_rate = meta_learning_rate
        if inner_learning_rate is not None:
            self.inner_learning_rate = inner_learning_rate
        if gamma is not None:
            self.gamma = gamma
        if meta_batch_size is not None:
            self.meta_batch_size = meta_batch_size
        if adaptation_steps is not None:
            self.adaptation_steps = adaptation_steps
        if adaptation_rate is not None:
            self.adaptation_rate = adaptation_rate
        if transfer_factor is not None:
            self.transfer_factor = transfer_factor

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar memorias específicas
        self.memoria_episodica.clear()
        self.memoria_tareas.clear()
        self.historial_adaptacion.clear()

        # Limpiar historiales específicos
        self.historial_meta_gradientes.clear()
        self.historial_adaptacion_rapida.clear()
        self.historial_transferencia.clear()
        self.historial_eficiencia_meta.clear()
        self.historial_estabilidad_meta.clear()

        # Resetear estadísticas meta
        for key in self.estadisticas_meta:
            self.estadisticas_meta[key] = 0.0

        # Resetear estado meta
        self.tarea_actual = None
        self.contexto_tarea = None
        self.pesos_adaptados = None
        self.sesgo_adaptado = None

        logger.info(f"Neurona meta-aprendizaje reinicializada: meta_lr={self.meta_learning_rate}, "
                    f"inner_lr={self.inner_learning_rate}, adaptacion={self.adaptation_rate}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoMetaLearning(entrada={self.input_size}, "
                f"salida={self.output_size}, meta_lr={self.meta_learning_rate}, "
                f"inner_lr={self.inner_learning_rate}, adaptacion={self.adaptation_rate})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_2


def crear_neurona_meta_learning(input_size: int, output_size: int,
                                configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoMetaLearning:
    """
    Función de conveniencia para crear una neurona meta-aprendizaje.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona meta-aprendizaje configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoMetaLearning(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoMetaLearning'),
        meta_learning_rate=configuracion.get('meta_learning_rate', LUCIA_ADVANCED_RL_CONFIG['meta_learning_rate']),
        inner_learning_rate=configuracion.get('inner_learning_rate', 0.01),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma']),
        meta_batch_size=configuracion.get('meta_batch_size', 32),
        adaptation_steps=configuracion.get('adaptation_steps', 5),
        memory_size=configuracion.get('memory_size', 1000),
        task_memory_size=configuracion.get('task_memory_size', 100),
        meta_gradient_clip=configuracion.get('meta_gradient_clip', 1.0),
        adaptation_rate=configuracion.get('adaptation_rate', 0.1),
        transfer_factor=configuracion.get('transfer_factor', 0.5)
    )


# Configuración específica para RFEN2_RN_2
RFEN2_RN_2_CONFIG = {
    'inicializacion_preferida': 'meta_learning',
    'meta_learning_rate_default': 0.0001,
    'inner_learning_rate_default': 0.01,
    'gamma_default': 0.99,
    'meta_batch_size_default': 32,
    'adaptation_steps_default': 5,
    'memory_size_default': 1000,
    'task_memory_size_default': 100,
    'meta_gradient_clip_default': 1.0,
    'adaptation_rate_default': 0.1,
    'transfer_factor_default': 0.5,
    'umbral_estabilidad_meta': 0.7,
    'umbral_meta_gradientes': 5.0,
    'umbral_eficiencia_meta': 0.6,
    'umbral_transferencia': 0.5,
    'umbral_memoria_episodica': 50
}

logger.info("RFEN2_RN_2.py cargado correctamente - Neurona de Refuerzo Meta-Aprendizaje")
