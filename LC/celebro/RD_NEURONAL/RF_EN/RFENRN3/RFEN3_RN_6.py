"""
RFEN3_RN_6 - Sistema de Consolidación de Memoria Inspirado en Sueño
Implementación de técnicas de consolidación de memoria basadas en procesos de sueño biológico
Incluye: Replay de experiencias, consolidación de pesos, y ciclos de sueño artificial
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import heapq
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SleepConsolidationConfig:
    """Configuración para consolidación de memoria inspirada en sueño"""
    sleep_frequency: int = 100  # Cada N pasos de entrenamiento
    sleep_duration: int = 10    # Número de ciclos de sueño
    replay_buffer_size: int = 10000
    replay_ratio: float = 0.1   # Proporción de datos para replay
    consolidation_strength: float = 0.5
    memory_decay_rate: float = 0.95
    importance_sampling: bool = True
    dream_generation: bool = True
    synaptic_consolidation: bool = True
    episodic_replay: bool = True


class ExperienceReplay:
    """
    Buffer de replay de experiencias con muestreo por importancia
    """

    def __init__(self, config: SleepConsolidationConfig):
        self.config = config
        self.buffer = deque(maxlen=config.replay_buffer_size)
        self.importance_scores = []
        self.experience_priorities = []
        self.beta = 0.4  # Parámetro para corrección de bias
        self.alpha = 0.6  # Parámetro para priorización

    def add_experience(self, state: torch.Tensor, action: torch.Tensor,
                       reward: float, next_state: torch.Tensor,
                       done: bool, td_error: float = None) -> None:
        """Añade una experiencia al buffer"""

        experience = {
            'state': state.clone(),
            'action': action.clone(),
            'reward': reward,
            'next_state': next_state.clone(),
            'done': done,
            'timestamp': time.time()
        }

        self.buffer.append(experience)

        # Calcular importancia basada en TD error
        if td_error is not None:
            priority = abs(td_error) + 1e-6
            self.experience_priorities.append(priority)

            if len(self.experience_priorities) > self.config.replay_buffer_size:
                self.experience_priorities.pop(0)

    def sample_batch(self, batch_size: int) -> Tuple[List[Dict], List[float]]:
        """Muestra un batch de experiencias con muestreo por importancia"""

        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if self.config.importance_sampling and len(self.experience_priorities) > 0:
            # Muestreo por importancia
            priorities = np.array(self.experience_priorities[-len(self.buffer):])
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            experiences = [self.buffer[i] for i in indices]

            # Calcular pesos de importancia
            weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
            weights /= weights.max()

        else:
            # Muestreo uniforme
            indices = np.random.choice(len(self.buffer), batch_size)
            experiences = [self.buffer[i] for i in indices]
            weights = np.ones(batch_size)

        return experiences, weights

    def get_recent_experiences(self, n: int) -> List[Dict]:
        """Obtiene las N experiencias más recientes"""
        return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)


class DreamGenerator:
    """
    Generador de sueños artificiales basado en patrones aprendidos
    """

    def __init__(self, config: SleepConsolidationConfig):
        self.config = config
        self.dream_patterns = []
        self.pattern_frequency = {}
        self.dream_memory = deque(maxlen=1000)

    def generate_dream(self, model: nn.Module,
                       real_experiences: List[Dict]) -> List[Dict]:
        """Genera sueños basados en experiencias reales"""

        dreams = []

        # Generar sueños basados en patrones frecuentes
        frequent_patterns = self._extract_frequent_patterns(real_experiences)

        for pattern in frequent_patterns[:5]:  # Top 5 patrones
            dream = self._create_dream_from_pattern(pattern, model)
            dreams.append(dream)

        # Generar sueños aleatorios basados en distribución de datos
        for _ in range(3):
            dream = self._generate_random_dream(model, real_experiences)
            dreams.append(dream)

        return dreams

    def _extract_frequent_patterns(self, experiences: List[Dict]) -> List[Dict]:
        """Extrae patrones frecuentes de las experiencias"""

        # Agrupar experiencias por características similares
        patterns = {}

        for exp in experiences:
            # Crear clave basada en características del estado
            state_key = self._create_state_key(exp['state'])

            if state_key not in patterns:
                patterns[state_key] = []
            patterns[state_key].append(exp)

        # Seleccionar patrones más frecuentes
        frequent_patterns = []
        for key, pattern_experiences in patterns.items():
            if len(pattern_experiences) >= 3:  # Mínimo 3 experiencias
                frequent_patterns.append({
                    'pattern_key': key,
                    'experiences': pattern_experiences,
                    'frequency': len(pattern_experiences)
                })

        # Ordenar por frecuencia
        frequent_patterns.sort(key=lambda x: x['frequency'], reverse=True)

        return frequent_patterns

    def _create_state_key(self, state: torch.Tensor) -> str:
        """Crea una clave para agrupar estados similares"""
        # Reducir dimensionalidad para crear clave
        if state.dim() > 1:
            state_summary = torch.mean(state, dim=tuple(range(1, state.dim())))
        else:
            state_summary = state

        # Discretizar para crear grupos
        discretized = torch.round(state_summary * 10) / 10
        return str(discretized.tolist())

    def _create_dream_from_pattern(self, pattern: Dict, model: nn.Module) -> Dict:
        """Crea un sueño basado en un patrón específico"""

        # Seleccionar experiencia base del patrón
        base_exp = random.choice(pattern['experiences'])

        # Modificar ligeramente para crear variación
        dream_state = base_exp['state'].clone()

        # Añadir ruido gaussiano
        noise = torch.randn_like(dream_state) * 0.1
        dream_state = dream_state + noise

        # Crear acción basada en el modelo
        with torch.no_grad():
            dream_action = model(dream_state)

        # Crear recompensa basada en el patrón
        dream_reward = base_exp['reward'] + random.gauss(0, 0.1)

        return {
            'state': dream_state,
            'action': dream_action,
            'reward': dream_reward,
            'next_state': dream_state.clone(),
            'done': False,
            'is_dream': True,
            'pattern_source': pattern['pattern_key']
        }

    def _generate_random_dream(self, model: nn.Module,
                               real_experiences: List[Dict]) -> Dict:
        """Genera un sueño completamente aleatorio"""

        if not real_experiences:
            return None

        # Usar estadísticas de experiencias reales
        states = [exp['state'] for exp in real_experiences]
        rewards = [exp['reward'] for exp in real_experiences]

        # Generar estado aleatorio basado en distribución
        state_mean = torch.mean(torch.stack(states), dim=0)
        state_std = torch.std(torch.stack(states), dim=0)

        dream_state = torch.normal(state_mean, state_std)

        # Generar acción
        with torch.no_grad():
            dream_action = model(dream_state)

        # Generar recompensa
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        dream_reward = np.random.normal(reward_mean, reward_std)

        return {
            'state': dream_state,
            'action': dream_action,
            'reward': dream_reward,
            'next_state': dream_state.clone(),
            'done': False,
            'is_dream': True,
            'pattern_source': 'random'
        }


class SynapticConsolidator:
    """
    Consolidador sináptico que simula procesos de consolidación durante el sueño
    """

    def __init__(self, config: SleepConsolidationConfig):
        self.config = config
        self.weight_history = []
        self.consolidation_strength = config.consolidation_strength
        self.memory_traces = {}

    def consolidate_weights(self, model: nn.Module,
                            experiences: List[Dict]) -> Dict[str, float]:
        """Consolida los pesos del modelo basado en experiencias"""

        consolidation_stats = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular traza de memoria para este parámetro
                memory_trace = self._calculate_memory_trace(param, experiences)

                # Aplicar consolidación
                consolidated_weight = self._apply_consolidation(param.data, memory_trace)

                # Actualizar peso
                param.data.copy_(consolidated_weight)

                # Registrar estadísticas
                consolidation_stats[name] = {
                    'memory_trace': memory_trace,
                    'weight_change': torch.norm(consolidated_weight - param.data).item(),
                    'consolidation_strength': self.consolidation_strength
                }

        return consolidation_stats

    def _calculate_memory_trace(self, param: torch.Tensor,
                                experiences: List[Dict]) -> torch.Tensor:
        """Calcula la traza de memoria para un parámetro"""

        # Inicializar traza
        trace = torch.zeros_like(param)

        # Acumular contribuciones de experiencias
        for exp in experiences:
            # Calcular importancia de la experiencia
            importance = self._calculate_experience_importance(exp)

            # Calcular contribución a la traza
            contribution = self._calculate_parameter_contribution(param, exp)

            # Acumular con peso de importancia
            trace += importance * contribution

        # Normalizar
        if len(experiences) > 0:
            trace /= len(experiences)

        return trace

    def _calculate_experience_importance(self, experience: Dict) -> float:
        """Calcula la importancia de una experiencia"""

        # Basado en recompensa y novedad
        reward_importance = abs(experience['reward'])

        # Basado en si es un sueño o experiencia real
        dream_factor = 0.5 if experience.get('is_dream', False) else 1.0

        return reward_importance * dream_factor

    def _calculate_parameter_contribution(self, param: torch.Tensor,
                                          experience: Dict) -> torch.Tensor:
        """Calcula la contribución de una experiencia a un parámetro"""

        # Simplificado: usar gradiente simulado
        state = experience['state']

        # Crear gradiente simulado basado en la experiencia
        if param.dim() == 2:  # Capa lineal
            # Simular gradiente basado en estado y acción
            grad_sim = torch.outer(state.flatten(), experience['action'].flatten())
            grad_sim = grad_sim[:param.size(0), :param.size(1)]
        else:
            # Para otros tipos de parámetros, usar gradiente aleatorio pequeño
            grad_sim = torch.randn_like(param) * 0.01

        return grad_sim

    def _apply_consolidation(self, weight: torch.Tensor,
                             memory_trace: torch.Tensor) -> torch.Tensor:
        """Aplica consolidación a los pesos"""

        # Consolidación basada en traza de memoria
        consolidated_weight = weight + self.consolidation_strength * memory_trace

        # Aplicar decaimiento de memoria
        consolidated_weight *= self.config.memory_decay_rate

        return consolidated_weight


class SleepCycleManager:
    """
    Gestor de ciclos de sueño que coordina la consolidación de memoria
    """

    def __init__(self, config: SleepConsolidationConfig):
        self.config = config
        self.experience_replay = ExperienceReplay(config)
        self.dream_generator = DreamGenerator(config)
        self.synaptic_consolidator = SynapticConsolidator(config)

        self.sleep_stats = {
            'total_sleep_cycles': 0,
            'experiences_replayed': 0,
            'dreams_generated': 0,
            'consolidation_events': 0
        }

        self.last_sleep_time = 0

    def should_sleep(self, current_step: int) -> bool:
        """Determina si es momento de dormir"""
        return current_step % self.config.sleep_frequency == 0

    def sleep_cycle(self, model: nn.Module, optimizer: optim.Optimizer,
                    loss_fn: Callable) -> Dict[str, float]:
        """Realiza un ciclo completo de sueño"""

        logger.info("Iniciando ciclo de sueño para consolidación de memoria")

        sleep_stats = {
            'consolidation_loss': 0.0,
            'replay_loss': 0.0,
            'dream_loss': 0.0,
            'weight_changes': 0.0
        }

        # 1. Replay de experiencias importantes
        if self.config.episodic_replay:
            replay_stats = self._replay_important_experiences(model, optimizer, loss_fn)
            sleep_stats.update(replay_stats)

        # 2. Generación y procesamiento de sueños
        if self.config.dream_generation:
            dream_stats = self._process_dreams(model, optimizer, loss_fn)
            sleep_stats.update(dream_stats)

        # 3. Consolidación sináptica
        if self.config.synaptic_consolidation:
            consolidation_stats = self._consolidate_synapses(model)
            sleep_stats.update(consolidation_stats)

        # Actualizar estadísticas
        self.sleep_stats['total_sleep_cycles'] += 1
        self.last_sleep_time = time.time()

        logger.info(f"Ciclo de sueño completado. Estadísticas: {sleep_stats}")

        return sleep_stats

    def _replay_important_experiences(self, model: nn.Module,
                                      optimizer: optim.Optimizer,
                                      loss_fn: Callable) -> Dict[str, float]:
        """Replay de experiencias importantes"""

        if len(self.experience_replay.buffer) < 10:
            return {'replay_loss': 0.0}

        # Muestrear experiencias importantes
        experiences, weights = self.experience_replay.sample_batch(32)

        total_loss = 0.0

        for exp, weight in zip(experiences, weights):
            optimizer.zero_grad()

            # Forward pass
            output = model(exp['state'])
            loss = loss_fn(output, exp['action'])

            # Aplicar peso de importancia
            weighted_loss = loss * weight

            # Backward pass
            weighted_loss.backward()
            optimizer.step()

            total_loss += loss.item()

        self.sleep_stats['experiences_replayed'] += len(experiences)

        return {'replay_loss': total_loss / len(experiences)}

    def _process_dreams(self, model: nn.Module, optimizer: optim.Optimizer,
                        loss_fn: Callable) -> Dict[str, float]:
        """Procesa sueños generados"""

        # Obtener experiencias recientes para generar sueños
        recent_experiences = self.experience_replay.get_recent_experiences(100)

        if not recent_experiences:
            return {'dream_loss': 0.0}

        # Generar sueños
        dreams = self.dream_generator.generate_dream(model, recent_experiences)

        total_loss = 0.0

        for dream in dreams:
            optimizer.zero_grad()

            # Forward pass con sueño
            output = model(dream['state'])
            loss = loss_fn(output, dream['action'])

            # Los sueños tienen menor peso
            dream_loss = loss * 0.5

            # Backward pass
            dream_loss.backward()
            optimizer.step()

            total_loss += loss.item()

        self.sleep_stats['dreams_generated'] += len(dreams)

        return {'dream_loss': total_loss / len(dreams) if dreams else 0.0}

    def _consolidate_synapses(self, model: nn.Module) -> Dict[str, float]:
        """Consolida sinapsis basado en experiencias"""

        # Obtener experiencias para consolidación
        experiences = self.experience_replay.get_recent_experiences(50)

        if not experiences:
            return {'weight_changes': 0.0}

        # Consolidar pesos
        consolidation_stats = self.synaptic_consolidator.consolidate_weights(model, experiences)

        # Calcular cambio total de pesos
        total_weight_change = sum(stats['weight_change'] for stats in consolidation_stats.values())

        self.sleep_stats['consolidation_events'] += 1

        return {'weight_changes': total_weight_change}

    def add_experience(self, state: torch.Tensor, action: torch.Tensor,
                       reward: float, next_state: torch.Tensor,
                       done: bool, td_error: float = None) -> None:
        """Añade una experiencia al buffer de replay"""
        self.experience_replay.add_experience(state, action, reward, next_state, done, td_error)

    def get_sleep_summary(self) -> Dict:
        """Obtiene un resumen del estado del sueño"""
        return {
            'config': {
                'sleep_frequency': self.config.sleep_frequency,
                'sleep_duration': self.config.sleep_duration,
                'replay_buffer_size': self.config.replay_buffer_size,
                'consolidation_strength': self.config.consolidation_strength
            },
            'stats': self.sleep_stats,
            'buffer_size': len(self.experience_replay.buffer),
            'last_sleep_time': self.last_sleep_time
        }

# Funciones de utilidad


def enable_sleep_consolidation(model: nn.Module,
                               optimizer: optim.Optimizer,
                               config: Optional[SleepConsolidationConfig] = None) -> SleepCycleManager:
    """Habilita consolidación de memoria inspirada en sueño"""

    if config is None:
        config = SleepConsolidationConfig()

    manager = SleepCycleManager(config)
    logger.info("Consolidación de memoria inspirada en sueño habilitada")

    return manager


def analyze_memory_consolidation(model: nn.Module,
                                 experiences: List[Dict]) -> Dict[str, float]:
    """Analiza la efectividad de la consolidación de memoria"""

    if not experiences:
        return {'status': 'no_experiences'}

    # Analizar distribución de experiencias
    rewards = [exp['reward'] for exp in experiences]
    states = [exp['state'] for exp in experiences]

    # Calcular métricas
    reward_variance = np.var(rewards)
    state_diversity = len(set(str(exp['state'].tolist()) for exp in experiences))

    # Analizar pesos del modelo
    weight_norms = []
    for param in model.parameters():
        if param.requires_grad:
            weight_norms.append(param.data.norm().item())

    return {
        'reward_variance': reward_variance,
        'state_diversity': state_diversity,
        'avg_weight_norm': np.mean(weight_norms),
        'weight_norm_std': np.std(weight_norms),
        'total_experiences': len(experiences),
        'consolidation_potential': reward_variance * state_diversity
    }


# Exportar clases y funciones principales
__all__ = [
    'SleepConsolidationConfig',
    'ExperienceReplay',
    'DreamGenerator',
    'SynapticConsolidator',
    'SleepCycleManager',
    'enable_sleep_consolidation',
    'analyze_memory_consolidation'
]

logger.info("RFEN3_RN_6 - Sistema de Consolidación de Memoria Inspirado en Sueño cargado correctamente")
