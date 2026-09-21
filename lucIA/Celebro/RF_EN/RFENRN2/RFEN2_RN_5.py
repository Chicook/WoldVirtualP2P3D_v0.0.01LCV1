"""
RFEN2_RN_5.py - Neurona de Refuerzo con Ciclos de Sueño Avanzados
================================================================

Esta neurona implementa Redes Neuronales con Ciclos de Sueño Avanzados
para consolidación de memoria y prevención del olvido catastrófico,
utilizando técnicas avanzadas de 2025 para optimización de pesos.

Características Avanzadas 2025:
- Ciclos de sueño artificiales con consolidación de memoria
- Prevención del olvido catastrófico mediante replay selectivo
- Consolidación de pesos durante estados de sueño profundo
- Replay de experiencias importantes durante el sueño REM
- Sincronización de memoria episódica y semántica
- Optimización de pesos durante ciclos de sueño
- Adaptación de ciclos según la carga de aprendizaje
- Consolidación selectiva de memorias críticas
- Astrocitos artificiales para gestión de señales temporales
- Controladores activos de neuronas durante el sueño

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
import random
from . import NeuronaRefuerzoAvanzadaBase, calcular_adaptacion_dinamica, \
    aplicar_consolidacion_memoria, LUCIA_ADVANCED_RL_CONFIG

logger = logging.getLogger('RFENRN2.RFEN2_RN_5')


class NeuronaRefuerzoCiclosSueno(NeuronaRefuerzoAvanzadaBase):
    """
    Neurona de refuerzo con Ciclos de Sueño Avanzados para consolidación de memoria.

    Esta neurona implementa ciclos de sueño artificiales que consolidan
    la memoria y previenen el olvido catastrófico mediante técnicas
    avanzadas de replay selectivo y consolidación de pesos.
    """

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaRefuerzoCiclosSueno",
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 sleep_cycle_length: int = 1000,
                 consolidation_rate: float = 0.1,
                 replay_ratio: float = 0.3,
                 memory_importance_threshold: float = 0.7,
                 sleep_intensity: float = 0.5,
                 consolidation_strength: float = 0.8,
                 forget_resistance: float = 0.9,
                 astrocyte_factor: float = 0.2,
                 active_controller_enabled: bool = True,
                 rem_sleep_ratio: float = 0.25,
                 deep_sleep_ratio: float = 0.4,
                 light_sleep_ratio: float = 0.35):
        """
        Inicializa la neurona de refuerzo con ciclos de sueño avanzados.

        Args:
            input_size: Número de características de entrada
            output_size: Número de acciones posibles
            nombre: Nombre identificativo de la neurona
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            sleep_cycle_length: Longitud del ciclo de sueño
            consolidation_rate: Tasa de consolidación
            replay_ratio: Proporción de replay durante el sueño
            memory_importance_threshold: Umbral de importancia de memoria
            sleep_intensity: Intensidad del sueño
            consolidation_strength: Fuerza de consolidación
            forget_resistance: Resistencia al olvido
            astrocyte_factor: Factor de astrocitos artificiales
            active_controller_enabled: Si habilitar controladores activos
            rem_sleep_ratio: Proporción de sueño REM
            deep_sleep_ratio: Proporción de sueño profundo
            light_sleep_ratio: Proporción de sueño ligero
        """
        super().__init__(input_size, output_size, nombre)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.sleep_cycle_length = sleep_cycle_length
        self.consolidation_rate = consolidation_rate
        self.replay_ratio = replay_ratio
        self.memory_importance_threshold = memory_importance_threshold
        self.sleep_intensity = sleep_intensity
        self.consolidation_strength = consolidation_strength
        self.forget_resistance = forget_resistance
        self.astrocyte_factor = astrocyte_factor
        self.active_controller_enabled = active_controller_enabled
        self.rem_sleep_ratio = rem_sleep_ratio
        self.deep_sleep_ratio = deep_sleep_ratio
        self.light_sleep_ratio = light_sleep_ratio

        # Pesos principales y pesos consolidados
        self.pesos_activos = None
        self.sesgo_activo = None
        self.pesos_consolidados = None
        self.sesgo_consolidado = None

        # Pesos de astrocitos artificiales
        self.pesos_astrocytes = None
        self.sesgo_astrocytes = None

        # Memoria episódica, semántica y crítica
        self.memoria_episodica = deque(maxlen=10000)
        self.memoria_semantica = deque(maxlen=5000)
        self.memoria_critica = deque(maxlen=1000)
        self.memoria_rem = deque(maxlen=2000)
        self.memoria_deep_sleep = deque(maxlen=3000)

        # Estado del ciclo de sueño
        self.estado_sueno = False
        self.fase_sueno = "despierto"  # despierto, light_sleep, deep_sleep, rem_sleep
        self.contador_ciclo = 0
        self.ultimo_sueno = 0
        self.duracion_fase_actual = 0

        # Controladores activos de neuronas
        self.controladores_activos = {}
        self.actividad_controladores = {}

        # Estadísticas específicas de ciclos de sueño avanzados
        self.estadisticas_sueno = {
            'ciclos_completados': 0,
            'memorias_consolidadas': 0,
            'replay_episodios': 0,
            'consolidacion_eficiencia': 0.0,
            'resistencia_olvido': 0.0,
            'intensidad_sueno_actual': 0.0,
            'memoria_episodica_size': 0,
            'memoria_semantica_size': 0,
            'memoria_critica_size': 0,
            'estabilidad_consolidacion': 0.0,
            'eficiencia_astrocytes': 0.0,
            'actividad_controladores': 0.0,
            'calidad_sueno_rem': 0.0,
            'calidad_sueno_profundo': 0.0,
            'calidad_sueno_ligero': 0.0,
            'sincronizacion_memoria': 0.0
        }

        # Historial de ciclos de sueño avanzados
        self.historial_ciclos_sueno = deque(maxlen=100)
        self.historial_consolidacion = deque(maxlen=1000)
        self.historial_replay = deque(maxlen=1000)
        self.historial_fases_sueno = deque(maxlen=1000)
        self.historial_astrocytes = deque(maxlen=1000)
        self.historial_controladores = deque(maxlen=1000)

        logger.info(f"NeuronaRefuerzoCiclosSueno creada: {self}")

    def inicializar_pesos(self) -> None:
        """
        Inicializa los pesos activos, consolidados y de astrocitos.
        """
        # Pesos activos (para aprendizaje activo)
        self.pesos_activos = np.random.normal(0, 0.1, (self.input_size, self.output_size))
        self.sesgo_activo = np.zeros((1, self.output_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Pesos consolidados (para memoria a largo plazo)
        self.pesos_consolidados = self.pesos_activos.copy()
        self.sesgo_consolidado = self.sesgo_activo.copy()

        # Pesos de astrocitos artificiales
        self.pesos_astrocytes = np.random.normal(0, 0.05, (self.input_size, self.input_size))
        self.sesgo_astrocytes = np.zeros((1, self.input_size), dtype=LUCIA_ADVANCED_RL_CONFIG['precision'])

        # Inicializar controladores activos
        if self.active_controller_enabled:
            for i in range(self.input_size):
                self.controladores_activos[f"controller_{i}"] = {
                    'peso': np.random.normal(0, 0.1),
                    'umbral': np.random.uniform(0.1, 0.9),
                    'actividad': 0.0,
                    'historial_actividad': deque(maxlen=100)
                }
                self.actividad_controladores[f"controller_{i}"] = 0.0

        logger.info("Pesos de ciclos de sueño avanzados inicializados")

    def _calcular_importancia_memoria_avanzada(self, estado: np.ndarray, accion: int,
                                               recompensa: float, contexto: Optional[np.ndarray] = None) -> float:
        """
        Calcula la importancia de una memoria usando técnicas avanzadas.

        Args:
            estado: Estado de la memoria
            accion: Acción tomada
            recompensa: Recompensa recibida
            contexto: Contexto adicional (opcional)

        Returns:
            Importancia de la memoria (0-1)
        """
        # Importancia basada en magnitud de recompensa
        importancia_recompensa = abs(recompensa) / (1.0 + abs(recompensa))

        # Importancia basada en rareza del estado
        if len(self.memoria_episodica) > 0:
            estados_similares = 0
            for memoria in self.memoria_episodica:
                similitud = np.dot(estado, memoria['estado']) / (
                    np.linalg.norm(estado) * np.linalg.norm(memoria['estado']) + 1e-8
                )
                if similitud > 0.8:
                    estados_similares += 1

            rareza = 1.0 / (1.0 + estados_similares)
        else:
            rareza = 1.0

        # Importancia basada en contexto (si está disponible)
        importancia_contexto = 1.0
        if contexto is not None:
            contexto_norm = np.linalg.norm(contexto)
            importancia_contexto = 1.0 + 0.1 * np.tanh(contexto_norm)

        # Importancia basada en actividad de controladores
        importancia_controladores = 1.0
        if self.active_controller_enabled:
            actividad_total = sum(self.actividad_controladores.values())
            importancia_controladores = 1.0 + 0.1 * actividad_total

        # Combinar factores
        importancia = (importancia_recompensa + rareza + importancia_contexto + importancia_controladores) / 4.0

        return min(importancia, 1.0)

    def _aplicar_astrocytes_artificiales(self, estado: np.ndarray) -> np.ndarray:
        """
        Aplica astrocitos artificiales para gestión de señales temporales.

        Args:
            estado: Estado actual

        Returns:
            Estado procesado por astrocitos
        """
        if self.pesos_astrocytes is None:
            return estado

        # Procesar estado con astrocitos
        estado_astrocytes = np.dot(estado, self.pesos_astrocytes) + self.sesgo_astrocytes

        # Aplicar función de activación de astrocitos (sigmoide suave)
        estado_astrocytes = np.tanh(estado_astrocytes)

        # Combinar con estado original
        estado_procesado = (1 - self.astrocyte_factor) * estado + self.astrocyte_factor * estado_astrocytes

        return estado_procesado

    def _actualizar_controladores_activos(self, estado: np.ndarray) -> None:
        """
        Actualiza los controladores activos de neuronas.

        Args:
            estado: Estado actual
        """
        if not self.active_controller_enabled:
            return

        for i, controller_id in enumerate(self.controladores_activos.keys()):
            controller = self.controladores_activos[controller_id]

            # Calcular actividad del controlador
            entrada_controlador = estado[i] if i < len(estado) else 0.0
            actividad = np.tanh(controller['peso'] * entrada_controlador)

            # Actualizar actividad
            controller['actividad'] = actividad
            self.actividad_controladores[controller_id] = actividad

            # Guardar en historial
            controller['historial_actividad'].append(actividad)

            # Adaptar umbral dinámicamente
            if len(controller['historial_actividad']) > 10:
                actividad_media = np.mean(list(controller['historial_actividad'])[-10:])
                controller['umbral'] = 0.9 * controller['umbral'] + 0.1 * actividad_media

    def almacenar_memoria_avanzada(self, estado: np.ndarray, accion: int, recompensa: float,
                                   siguiente_estado: np.ndarray, terminado: bool,
                                   contexto: Optional[np.ndarray] = None) -> None:
        """
        Almacena una memoria con evaluación avanzada de importancia.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            terminado: Si el episodio terminó
            contexto: Contexto adicional (opcional)
        """
        # Procesar estado con astrocitos
        estado_procesado = self._aplicar_astrocytes_artificiales(estado)

        # Actualizar controladores activos
        self._actualizar_controladores_activos(estado)

        # Calcular importancia avanzada
        importancia = self._calcular_importancia_memoria_avanzada(estado_procesado, accion, recompensa, contexto)

        memoria = {
            'estado': estado_procesado.copy(),
            'estado_original': estado.copy(),
            'accion': accion,
            'recompensa': recompensa,
            'siguiente_estado': siguiente_estado.copy(),
            'terminado': terminado,
            'importancia': importancia,
            'contexto': contexto.copy() if contexto is not None else None,
            'actividad_controladores': self.actividad_controladores.copy(),
            'timestamp': time.time()
        }

        # Almacenar en memoria episódica
        self.memoria_episodica.append(memoria)

        # Clasificar según importancia
        if importancia > self.memory_importance_threshold:
            self.memoria_critica.append(memoria)

        # Almacenar en memoria semántica si es significativa
        if abs(recompensa) > 0.5:
            self.memoria_semantica.append(memoria)

        # Actualizar estadísticas
        self.estadisticas_sueno['memoria_episodica_size'] = len(self.memoria_episodica)
        self.estadisticas_sueno['memoria_semantica_size'] = len(self.memoria_semantica)
        self.estadisticas_sueno['memoria_critica_size'] = len(self.memoria_critica)

    def _determinar_fase_sueno(self) -> str:
        """
        Determina la fase actual del sueño basada en el ciclo.

        Returns:
            Fase actual del sueño
        """
        ciclo_normalizado = (self.contador_ciclo % self.sleep_cycle_length) / self.sleep_cycle_length

        if ciclo_normalizado < self.light_sleep_ratio:
            return "light_sleep"
        elif ciclo_normalizado < self.light_sleep_ratio + self.deep_sleep_ratio:
            return "deep_sleep"
        else:
            return "rem_sleep"

    def _consolidar_memoria_fase_especifica(self, fase: str) -> None:
        """
        Consolida la memoria según la fase específica del sueño.

        Args:
            fase: Fase actual del sueño
        """
        if len(self.memoria_episodica) < 10:
            return

        # Seleccionar memorias según la fase
        if fase == "light_sleep":
            memorias_consolidar = list(self.memoria_episodica)[-50:]  # Memorias recientes
            intensidad_consolidacion = 0.3
        elif fase == "deep_sleep":
            memorias_consolidar = list(self.memoria_critica)[-30:]  # Memorias críticas
            intensidad_consolidacion = 0.8
        elif fase == "rem_sleep":
            memorias_consolidar = list(self.memoria_semantica)[-40:]  # Memorias semánticas
            intensidad_consolidacion = 0.6
        else:
            memorias_consolidar = list(self.memoria_episodica)[-20:]
            intensidad_consolidacion = 0.5

        if not memorias_consolidar:
            return

        # Consolidar pesos basado en memorias de la fase
        pesos_consolidacion = np.zeros_like(self.pesos_activos)
        sesgo_consolidacion = np.zeros_like(self.sesgo_activo)

        for memoria in memorias_consolidar:
            # Calcular contribución de esta memoria
            contribucion = memoria['importancia'] * self.consolidation_strength * intensidad_consolidacion

            # Aproximación de gradiente basado en la memoria
            estado = memoria['estado']
            accion = memoria['accion']
            recompensa = memoria['recompensa']

            # Calcular probabilidades actuales
            logits = np.dot(estado, self.pesos_activos) + self.sesgo_activo
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Gradiente de política
            grad_log_prob = np.zeros(self.output_size)
            grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

            # Aplicar contribución
            pesos_consolidacion += contribucion * np.outer(estado, grad_log_prob) * recompensa
            sesgo_consolidacion += contribucion * grad_log_prob * recompensa

        # Normalizar contribuciones
        n_memorias = len(memorias_consolidar)
        if n_memorias > 0:
            pesos_consolidacion /= n_memorias
            sesgo_consolidacion /= n_memorias

        # Actualizar pesos consolidados según la fase
        if fase == "deep_sleep":
            # Consolidación más fuerte en sueño profundo
            self.pesos_consolidados = (1 - self.consolidation_rate * 2) * self.pesos_consolidados + \
                self.consolidation_rate * 2 * (self.pesos_activos + pesos_consolidacion)
            self.sesgo_consolidado = (1 - self.consolidation_rate * 2) * self.sesgo_consolidado + \
                self.consolidation_rate * 2 * (self.sesgo_activo + sesgo_consolidacion)
        else:
            # Consolidación normal en otras fases
            self.pesos_consolidados = (1 - self.consolidation_rate) * self.pesos_consolidados + \
                self.consolidation_rate * (self.pesos_activos + pesos_consolidacion)
            self.sesgo_consolidado = (1 - self.consolidation_rate) * self.sesgo_consolidado + \
                self.consolidation_rate * (self.sesgo_activo + sesgo_consolidacion)

        # Actualizar estadísticas específicas de la fase
        if fase == "light_sleep":
            self.estadisticas_sueno['calidad_sueno_ligero'] = intensidad_consolidacion
        elif fase == "deep_sleep":
            self.estadisticas_sueno['calidad_sueno_profundo'] = intensidad_consolidacion
        elif fase == "rem_sleep":
            self.estadisticas_sueno['calidad_sueno_rem'] = intensidad_consolidacion

        # Guardar en historial de consolidación
        self.historial_consolidacion.append({
            'fase': fase,
            'memorias_consolidadas': n_memorias,
            'intensidad': intensidad_consolidacion,
            'timestamp': time.time()
        })

    def _replay_memorias_fase_especifica(self, fase: str) -> None:
        """
        Realiza replay de memorias según la fase específica del sueño.

        Args:
            fase: Fase actual del sueño
        """
        if len(self.memoria_episodica) < 10:
            return

        # Seleccionar memorias para replay según la fase
        if fase == "light_sleep":
            memorias_replay = random.sample(list(self.memoria_episodica),
                                            min(int(len(self.memoria_episodica) * self.replay_ratio * 0.5),
                                                len(self.memoria_episodica)))
            intensidad_replay = 0.3
        elif fase == "deep_sleep":
            memorias_replay = random.sample(list(self.memoria_critica),
                                            min(int(len(self.memoria_critica) * self.replay_ratio),
                                                len(self.memoria_critica)))
            intensidad_replay = 0.8
        elif fase == "rem_sleep":
            memorias_replay = random.sample(list(self.memoria_semantica),
                                            min(int(len(self.memoria_semantica) * self.replay_ratio * 0.7),
                                                len(self.memoria_semantica)))
            intensidad_replay = 0.6
        else:
            memorias_replay = random.sample(list(self.memoria_episodica),
                                            min(int(len(self.memoria_episodica) * self.replay_ratio * 0.3),
                                                len(self.memoria_episodica)))
            intensidad_replay = 0.4

        # Realizar replay
        for memoria in memorias_replay:
            estado = memoria['estado']
            accion = memoria['accion']
            recompensa = memoria['recompensa']

            # Calcular gradiente de replay
            logits = np.dot(estado, self.pesos_activos) + self.sesgo_activo
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Gradiente de política
            grad_log_prob = np.zeros(self.output_size)
            grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

            # Actualizar pesos activos con replay
            grad_pesos = np.outer(estado, grad_log_prob) * recompensa
            grad_sesgo = grad_log_prob * recompensa

            self.pesos_activos -= self.learning_rate * self.sleep_intensity * intensidad_replay * grad_pesos
            self.sesgo_activo -= self.learning_rate * self.sleep_intensity * intensidad_replay * grad_sesgo

        # Actualizar estadísticas
        self.estadisticas_sueno['replay_episodios'] += len(memorias_replay)

        # Guardar en historial de replay
        self.historial_replay.append({
            'fase': fase,
            'episodios_replay': len(memorias_replay),
            'intensidad': intensidad_replay,
            'timestamp': time.time()
        })

    def _iniciar_ciclo_sueno_avanzado(self) -> None:
        """
        Inicia un ciclo de sueño avanzado con múltiples fases.
        """
        self.estado_sueno = True
        self.contador_ciclo = 0
        self.ultimo_sueno = time.time()
        self.fase_sueno = "light_sleep"
        self.duracion_fase_actual = 0

        logger.info("Iniciando ciclo de sueño avanzado para consolidación de memoria")

    def _procesar_fase_sueno(self) -> None:
        """
        Procesa la fase actual del sueño.
        """
        if not self.estado_sueno:
            return

        # Determinar fase actual
        fase_actual = self._determinar_fase_sueno()

        # Cambiar fase si es necesario
        if fase_actual != self.fase_sueno:
            self.fase_sueno = fase_actual
            self.duracion_fase_actual = 0
            logger.info(f"Cambiando a fase de sueño: {fase_actual}")

        # Consolidar memoria según la fase
        self._consolidar_memoria_fase_especifica(fase_actual)

        # Replay de memorias según la fase
        self._replay_memorias_fase_especifica(fase_actual)

        # Actualizar duración de fase
        self.duracion_fase_actual += 1

        # Guardar en historial de fases
        self.historial_fases_sueno.append({
            'fase': fase_actual,
            'duracion': self.duracion_fase_actual,
            'timestamp': time.time()
        })

    def _finalizar_ciclo_sueno_avanzado(self) -> None:
        """
        Finaliza el ciclo de sueño avanzado y actualiza estadísticas.
        """
        self.estado_sueno = False
        self.fase_sueno = "despierto"
        self.estadisticas_sueno['ciclos_completados'] += 1

        # Calcular eficiencia de consolidación
        if len(self.historial_consolidacion) > 0:
            consolidaciones_recientes = [h['memorias_consolidadas'] for h in
                                         list(self.historial_consolidacion)[-10:]]
            self.estadisticas_sueno['consolidacion_eficiencia'] = np.mean(consolidaciones_recientes)

        # Calcular resistencia al olvido
        if len(self.historial_pesos) > 0:
            pesos_recientes = np.array(self.historial_pesos[-10:])
            varianza_pesos = np.var(pesos_recientes)
            self.estadisticas_sueno['resistencia_olvido'] = 1.0 / (1.0 + varianza_pesos)

        # Calcular eficiencia de astrocitos
        if self.pesos_astrocytes is not None:
            actividad_astrocytes = np.mean(np.abs(self.pesos_astrocytes))
            self.estadisticas_sueno['eficiencia_astrocytes'] = actividad_astrocytes

        # Calcular actividad de controladores
        if self.active_controller_enabled:
            actividad_total = sum(self.actividad_controladores.values())
            self.estadisticas_sueno['actividad_controladores'] = actividad_total

        # Calcular sincronización de memoria
        if len(self.memoria_episodica) > 0 and len(self.memoria_semantica) > 0:
            # Sincronización basada en correlación entre memorias
            episodicas_recientes = [m['importancia'] for m in list(self.memoria_episodica)[-100:]]
            semanticas_recientes = [m['importancia'] for m in list(self.memoria_semantica)[-100:]]

            if len(episodicas_recientes) > 0 and len(semanticas_recientes) > 0:
                correlacion = np.corrcoef(episodicas_recientes, semanticas_recientes)[0, 1]
                self.estadisticas_sueno['sincronizacion_memoria'] = abs(correlacion) if not np.isnan(correlacion) else 0.0

        # Guardar ciclo de sueño
        self.historial_ciclos_sueno.append({
            'duracion': time.time() - self.ultimo_sueno,
            'fases_completadas': len(set([h['fase'] for h in list(self.historial_fases_sueno)[-100:]])),
            'memorias_consolidadas': self.estadisticas_sueno['memorias_consolidadas'],
            'replay_episodios': self.estadisticas_sueno['replay_episodios'],
            'timestamp': time.time()
        })

        logger.info("Ciclo de sueño avanzado completado")

    def _verificar_ciclo_sueno_avanzado(self) -> None:
        """
        Verifica si es necesario iniciar un ciclo de sueño avanzado.
        """
        if self.contador_ciclo >= self.sleep_cycle_length:
            self._iniciar_ciclo_sueno_avanzado()

            # Procesar todas las fases del sueño
            for _ in range(int(self.sleep_cycle_length * 0.1)):  # Simular procesamiento
                self._procesar_fase_sueno()
                self.contador_ciclo += 1

            # Finalizar ciclo
            self._finalizar_ciclo_sueno_avanzado()

            # Resetear contador
            self.contador_ciclo = 0

    def forward(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Propagación hacia adelante con consolidación de memoria avanzada.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción o valor estimado
        """
        if self.pesos_activos is None:
            self.inicializar_pesos()

        # Incrementar contador de ciclo
        self.contador_ciclo += 1

        # Verificar si es necesario un ciclo de sueño
        self._verificar_ciclo_sueno_avanzado()

        # Procesar estado con astrocitos
        estado_procesado = self._aplicar_astrocytes_artificiales(estado)

        # Actualizar controladores activos
        self._actualizar_controladores_activos(estado_procesado)

        # Usar pesos activos durante el aprendizaje activo
        if not self.estado_sueno:
            pesos_actuales = self.pesos_activos
            sesgo_actual = self.sesgo_activo
        else:
            # Durante el sueño, usar pesos consolidados
            pesos_actuales = self.pesos_consolidados
            sesgo_actual = self.sesgo_consolidado

        # Calcular salida
        logits = np.dot(estado_procesado, pesos_actuales) + sesgo_actual

        # Aplicar función de activación
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probabilidades

    def seleccionar_accion(self, estado: np.ndarray, contexto: Optional[np.ndarray] = None) -> int:
        """
        Selecciona una acción usando la política con consolidación de memoria avanzada.

        Args:
            estado: Estado actual del entorno
            contexto: Contexto adicional (opcional)

        Returns:
            Acción seleccionada
        """
        probabilidades = self.forward(estado, contexto)

        # Selección basada en probabilidades
        accion = np.random.choice(self.output_size, p=probabilidades[0])

        return accion

    def entrenar_paso_avanzado(self, estado: np.ndarray, accion: int, recompensa: float,
                               siguiente_estado: np.ndarray, terminado: bool,
                               contexto: Optional[np.ndarray] = None) -> None:
        """
        Realiza un paso de entrenamiento con almacenamiento de memoria avanzada.

        Args:
            estado: Estado actual
            accion: Acción tomada
            recompensa: Recompensa recibida
            siguiente_estado: Siguiente estado
            terminado: Si el episodio terminó
            contexto: Contexto adicional (opcional)
        """
        # Almacenar memoria avanzada
        self.almacenar_memoria_avanzada(estado, accion, recompensa, siguiente_estado, terminado, contexto)

        # Entrenamiento activo (solo si no está en sueño)
        if not self.estado_sueno:
            # Procesar estado con astrocitos
            estado_procesado = self._aplicar_astrocytes_artificiales(estado)

            # Calcular gradiente
            logits = np.dot(estado_procesado, self.pesos_activos) + self.sesgo_activo
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Gradiente de política
            grad_log_prob = np.zeros(self.output_size)
            grad_log_prob[accion] = 1.0 / (probabilidades[0, accion] + 1e-8)

            # Actualizar pesos activos
            grad_pesos = np.outer(estado_procesado, grad_log_prob) * recompensa
            grad_sesgo = grad_log_prob * recompensa

            self.pesos_activos -= self.learning_rate * grad_pesos
            self.sesgo_activo -= self.learning_rate * grad_sesgo

            # Guardar en historial de pesos
            self.historial_pesos.append(self.pesos_activos.copy())

    def obtener_estadisticas_sueno_avanzadas(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas específicas de los ciclos de sueño avanzados.

        Returns:
            Diccionario con estadísticas de sueño avanzadas
        """
        if self.pesos_activos is None:
            return {'estado': 'no_inicializada'}

        stats_sueno = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'sleep_cycle_length': self.sleep_cycle_length,
            'consolidation_rate': self.consolidation_rate,
            'replay_ratio': self.replay_ratio,
            'memory_importance_threshold': self.memory_importance_threshold,
            'sleep_intensity': self.sleep_intensity,
            'consolidation_strength': self.consolidation_strength,
            'forget_resistance': self.forget_resistance,
            'astrocyte_factor': self.astrocyte_factor,
            'active_controller_enabled': self.active_controller_enabled,
            'rem_sleep_ratio': self.rem_sleep_ratio,
            'deep_sleep_ratio': self.deep_sleep_ratio,
            'light_sleep_ratio': self.light_sleep_ratio,
            'estado_sueno': self.estado_sueno,
            'fase_sueno': self.fase_sueno,
            'contador_ciclo': self.contador_ciclo,
            'ciclos_completados': self.estadisticas_sueno['ciclos_completados'],
            'memorias_consolidadas': self.estadisticas_sueno['memorias_consolidadas'],
            'replay_episodios': self.estadisticas_sueno['replay_episodios'],
            'consolidacion_eficiencia': self.estadisticas_sueno['consolidacion_eficiencia'],
            'resistencia_olvido': self.estadisticas_sueno['resistencia_olvido'],
            'memoria_episodica_size': self.estadisticas_sueno['memoria_episodica_size'],
            'memoria_semantica_size': self.estadisticas_sueno['memoria_semantica_size'],
            'memoria_critica_size': self.estadisticas_sueno['memoria_critica_size'],
            'estabilidad_consolidacion': self.estadisticas_sueno['estabilidad_consolidacion'],
            'eficiencia_astrocytes': self.estadisticas_sueno['eficiencia_astrocytes'],
            'actividad_controladores': self.estadisticas_sueno['actividad_controladores'],
            'calidad_sueno_rem': self.estadisticas_sueno['calidad_sueno_rem'],
            'calidad_sueno_profundo': self.estadisticas_sueno['calidad_sueno_profundo'],
            'calidad_sueno_ligero': self.estadisticas_sueno['calidad_sueno_ligero'],
            'sincronizacion_memoria': self.estadisticas_sueno['sincronizacion_memoria']
        }

        return stats_sueno

    def verificar_estabilidad(self) -> Dict[str, bool]:
        """
        Verifica la estabilidad de la neurona con ciclos de sueño avanzados.

        Returns:
            Diccionario con indicadores de estabilidad
        """
        estabilidad = super().verificar_estabilidad_avanzada()

        # Verificaciones específicas de ciclos de sueño avanzados
        estabilidad['consolidacion_eficiente'] = self.estadisticas_sueno['consolidacion_eficiencia'] > 10
        estabilidad['resistencia_olvido_ok'] = self.estadisticas_sueno['resistencia_olvido'] > 0.7
        estabilidad['memoria_episodica_suficiente'] = self.estadisticas_sueno['memoria_episodica_size'] > 100
        estabilidad['memoria_critica_balanceada'] = self.estadisticas_sueno['memoria_critica_size'] > 10
        estabilidad['ciclos_sueno_regulares'] = self.estadisticas_sueno['ciclos_completados'] > 0
        estabilidad['astrocytes_funcionales'] = self.estadisticas_sueno['eficiencia_astrocytes'] > 0.01
        estabilidad['controladores_activos'] = self.estadisticas_sueno['actividad_controladores'] > 0.0
        estabilidad['sincronizacion_memoria_ok'] = self.estadisticas_sueno['sincronizacion_memoria'] > 0.3

        return estabilidad

    def reinicializar_con_parametros(self, learning_rate: float = None,
                                     gamma: float = None,
                                     sleep_cycle_length: int = None,
                                     consolidation_rate: float = None,
                                     replay_ratio: float = None,
                                     memory_importance_threshold: float = None,
                                     sleep_intensity: float = None,
                                     consolidation_strength: float = None,
                                     forget_resistance: float = None,
                                     astrocyte_factor: float = None,
                                     active_controller_enabled: bool = None,
                                     rem_sleep_ratio: float = None,
                                     deep_sleep_ratio: float = None,
                                     light_sleep_ratio: float = None) -> None:
        """
        Reinicializa la neurona con ciclos de sueño avanzados con nuevos parámetros.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gamma is not None:
            self.gamma = gamma
        if sleep_cycle_length is not None:
            self.sleep_cycle_length = sleep_cycle_length
        if consolidation_rate is not None:
            self.consolidation_rate = consolidation_rate
        if replay_ratio is not None:
            self.replay_ratio = replay_ratio
        if memory_importance_threshold is not None:
            self.memory_importance_threshold = memory_importance_threshold
        if sleep_intensity is not None:
            self.sleep_intensity = sleep_intensity
        if consolidation_strength is not None:
            self.consolidation_strength = consolidation_strength
        if forget_resistance is not None:
            self.forget_resistance = forget_resistance
        if astrocyte_factor is not None:
            self.astrocyte_factor = astrocyte_factor
        if active_controller_enabled is not None:
            self.active_controller_enabled = active_controller_enabled
        if rem_sleep_ratio is not None:
            self.rem_sleep_ratio = rem_sleep_ratio
        if deep_sleep_ratio is not None:
            self.deep_sleep_ratio = deep_sleep_ratio
        if light_sleep_ratio is not None:
            self.light_sleep_ratio = light_sleep_ratio

        self.inicializar_pesos()
        self.resetear_historial()

        # Limpiar memorias avanzadas
        self.memoria_episodica.clear()
        self.memoria_semantica.clear()
        self.memoria_critica.clear()
        self.memoria_rem.clear()
        self.memoria_deep_sleep.clear()

        # Limpiar historiales específicos
        self.historial_ciclos_sueno.clear()
        self.historial_consolidacion.clear()
        self.historial_replay.clear()
        self.historial_fases_sueno.clear()
        self.historial_astrocytes.clear()
        self.historial_controladores.clear()

        # Resetear estado de sueño avanzado
        self.estado_sueno = False
        self.fase_sueno = "despierto"
        self.contador_ciclo = 0
        self.ultimo_sueno = 0
        self.duracion_fase_actual = 0

        # Resetear controladores activos
        if self.active_controller_enabled:
            for controller_id in self.controladores_activos.keys():
                self.controladores_activos[controller_id]['historial_actividad'].clear()
                self.actividad_controladores[controller_id] = 0.0

        # Resetear estadísticas de sueño avanzadas
        for key in self.estadisticas_sueno:
            self.estadisticas_sueno[key] = 0.0

        logger.info(f"Neurona ciclos de sueño avanzados reinicializada: lr={self.learning_rate}, "
                    f"ciclo_length={self.sleep_cycle_length}, consolidation={self.consolidation_rate}, "
                    f"astrocytes={self.astrocyte_factor}, controladores={self.active_controller_enabled}")

    def __str__(self) -> str:
        return (f"NeuronaRefuerzoCiclosSueno(entrada={self.input_size}, "
                f"salida={self.output_size}, lr={self.learning_rate}, "
                f"ciclo_length={self.sleep_cycle_length}, consolidation={self.consolidation_rate}, "
                f"astrocytes={self.astrocyte_factor}, controladores={self.active_controller_enabled})")

    def __repr__(self) -> str:
        return self.__str__()

# Funciones de utilidad específicas para RFEN2_RN_5


def crear_neurona_ciclos_sueno_avanzados(input_size: int, output_size: int,
                                         configuracion: Dict[str, Any] = None) -> NeuronaRefuerzoCiclosSueno:
    """
    Función de conveniencia para crear una neurona con ciclos de sueño avanzados.

    Args:
        input_size: Tamaño de entrada
        output_size: Tamaño de salida
        configuracion: Configuración personalizada

    Returns:
        Neurona con ciclos de sueño avanzados configurada
    """
    if configuracion is None:
        configuracion = {}

    return NeuronaRefuerzoCiclosSueno(
        input_size=input_size,
        output_size=output_size,
        nombre=configuracion.get('nombre', 'NeuronaRefuerzoCiclosSueno'),
        learning_rate=configuracion.get('learning_rate', LUCIA_ADVANCED_RL_CONFIG['default_learning_rate']),
        gamma=configuracion.get('gamma', LUCIA_ADVANCED_RL_CONFIG['default_gamma']),
        sleep_cycle_length=configuracion.get('sleep_cycle_length', LUCIA_ADVANCED_RL_CONFIG['sleep_cycle_length']),
        consolidation_rate=configuracion.get('consolidation_rate', 0.1),
        replay_ratio=configuracion.get('replay_ratio', 0.3),
        memory_importance_threshold=configuracion.get('memory_importance_threshold', 0.7),
        sleep_intensity=configuracion.get('sleep_intensity', 0.5),
        consolidation_strength=configuracion.get('consolidation_strength', 0.8),
        forget_resistance=configuracion.get('forget_resistance', 0.9),
        astrocyte_factor=configuracion.get('astrocyte_factor', 0.2),
        active_controller_enabled=configuracion.get('active_controller_enabled', True),
        rem_sleep_ratio=configuracion.get('rem_sleep_ratio', 0.25),
        deep_sleep_ratio=configuracion.get('deep_sleep_ratio', 0.4),
        light_sleep_ratio=configuracion.get('light_sleep_ratio', 0.35)
    )


# Configuración específica para RFEN2_RN_5
RFEN2_RN_5_CONFIG = {
    'inicializacion_preferida': 'ciclos_sueno_avanzados',
    'learning_rate_default': 0.0001,
    'gamma_default': 0.99,
    'sleep_cycle_length_default': 1000,
    'consolidation_rate_default': 0.1,
    'replay_ratio_default': 0.3,
    'memory_importance_threshold_default': 0.7,
    'sleep_intensity_default': 0.5,
    'consolidation_strength_default': 0.8,
    'forget_resistance_default': 0.9,
    'astrocyte_factor_default': 0.2,
    'active_controller_enabled_default': True,
    'rem_sleep_ratio_default': 0.25,
    'deep_sleep_ratio_default': 0.4,
    'light_sleep_ratio_default': 0.35,
    'umbral_consolidacion_eficiente': 10,
    'umbral_resistencia_olvido': 0.7,
    'umbral_memoria_episodica': 100,
    'umbral_memoria_critica': 10,
    'umbral_ciclos_sueno': 0,
    'umbral_astrocytes_funcionales': 0.01,
    'umbral_controladores_activos': 0.0,
    'umbral_sincronizacion_memoria': 0.3
}

logger.info("RFEN2_RN_5.py cargado correctamente - Neurona de Refuerzo Ciclos de Sueño Avanzados")
