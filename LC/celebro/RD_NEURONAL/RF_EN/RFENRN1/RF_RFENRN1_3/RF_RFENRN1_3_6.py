"""
RF_RFENRN1_3_6.py - Gestor de Sincronización de Pesos
======================================================

Implementa técnicas avanzadas de sincronización de pesos para entrenamiento
distribuido de redes neuronales de aprendizaje por refuerzo. Incluye
sincronización automática, detección de desincronización, corrección de
pesos divergentes y técnicas de consensus para mantener coherencia.

Características:
- Sincronización automática de pesos en entrenamiento distribuido
- Detección de desincronización entre procesos
- Corrección automática de pesos divergentes
- Técnicas de consensus para pesos coherentes
- Sincronización adaptativa basada en latencia
- Validación de coherencia de pesos
- Recuperación automática de sincronización

Autor: LucIA Development Team
Versión: 3.0.0
"""

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
except ImportError:
    pass  # dependencia pesada opcional
try:
    torch
except NameError:
    import types as _t
    torch = _t.SimpleNamespace(
        no_grad=lambda *a, **k: (lambda f: f) if a and callable(a[0]) else (lambda f: f),
        optim=_t.SimpleNamespace(Optimizer=object),
        Tensor=object,
    )
try:
    nn
except NameError:
    import types as _t2
    nn = _t2.SimpleNamespace(Module=object)
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger('RFENRN1.RF_RFENRN1_3_6')


@dataclass
class WeightSynchronizationConfig:
    """Configuración para sincronización de pesos"""
    use_automatic_sync: bool = True
    use_divergence_detection: bool = True
    use_consensus_sync: bool = True
    sync_frequency: int = 10
    divergence_threshold: float = 0.1
    consensus_threshold: float = 0.05
    max_sync_iterations: int = 3
    sync_timeout: float = 30.0
    use_adaptive_sync: bool = True
    latency_threshold: float = 1.0


class WeightSynchronizationManager:
    """
    Gestor de sincronización de pesos para entrenamiento distribuido.

    Implementa técnicas avanzadas de sincronización que mantienen
    coherencia de pesos en entrenamiento distribuido.
    """

    def __init__(self, config: Optional[WeightSynchronizationConfig] = None):
        """
        Inicializa el gestor de sincronización.

        Args:
            config: Configuración de sincronización (opcional)
        """
        self.config = config or WeightSynchronizationConfig()
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'divergence_detections': 0,
            'consensus_iterations': 0,
            'sync_latency': 0.0,
            'coherence_score': 0.0
        }
        self.weight_history = defaultdict(list)
        self.sync_history = []

        logger.info("WeightSynchronizationManager inicializado")

    def synchronize_model_weights(self, model: nn.Module, step: int) -> nn.Module:
        """
        Sincroniza pesos del modelo.

        Args:
            model: Modelo PyTorch
            step: Paso actual

        Returns:
            Modelo con pesos sincronizados
        """
        if not dist.is_initialized():
            return model

        if step % self.config.sync_frequency != 0:
            return model

        start_time = time.time()

        try:
            # Detectar divergencia
            if self.config.use_divergence_detection:
                divergence_detected = self._detect_divergence(model)
                if divergence_detected:
                    logger.warning("Divergencia de pesos detectada, aplicando corrección")
                    model = self._correct_divergence(model)

            # Sincronización automática
            if self.config.use_automatic_sync:
                model = self._automatic_synchronization(model)

            # Sincronización por consensus
            if self.config.use_consensus_sync:
                model = self._consensus_synchronization(model)

            # Actualizar estadísticas
            sync_time = time.time() - start_time
            self.sync_stats['sync_latency'] = sync_time
            self.sync_stats['successful_syncs'] += 1

            self.sync_history.append({
                'step': step,
                'time': sync_time,
                'success': True
            })

        except Exception as e:
            logger.error(f"Error en sincronización: {e}")
            self.sync_stats['failed_syncs'] += 1

            self.sync_history.append({
                'step': step,
                'time': time.time() - start_time,
                'success': False,
                'error': str(e)
            })

        self.sync_stats['total_syncs'] += 1

        return model

    def _detect_divergence(self, model: nn.Module) -> bool:
        """
        Detecta divergencia en pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            True si se detecta divergencia
        """
        divergence_detected = False

        for name, param in model.named_parameters():
            if 'weight' in name:
                # Calcular norma del peso
                weight_norm = torch.norm(param.data).item()

                # Comparar con historial
                if name in self.weight_history:
                    if len(self.weight_history[name]) >= 3:
                        historical_norms = self.weight_history[name][-3:]
                        mean_historical = np.mean(historical_norms)

                        if mean_historical > 0:
                            deviation = abs(weight_norm - mean_historical) / mean_historical

                            if deviation > self.config.divergence_threshold:
                                divergence_detected = True
                                logger.warning(f"Divergencia detectada en {name}: {deviation:.4f}")

                # Actualizar historial
                self.weight_history[name].append(weight_norm)
                if len(self.weight_history[name]) > 10:
                    self.weight_history[name].pop(0)

        if divergence_detected:
            self.sync_stats['divergence_detections'] += 1

        return divergence_detected

    def _correct_divergence(self, model: nn.Module) -> nn.Module:
        """
        Corrige divergencia en pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con divergencia corregida
        """
        for name, param in model.named_parameters():
            if 'weight' in name and name in self.weight_history:
                if len(self.weight_history[name]) >= 3:
                    historical_norms = self.weight_history[name][-3:]
                    target_norm = np.mean(historical_norms)

                    current_norm = torch.norm(param.data).item()

                    if current_norm > 0 and target_norm > 0:
                        # Corregir hacia la norma objetivo
                        correction_factor = target_norm / current_norm
                        param.data *= correction_factor

                        logger.info(f"Divergencia corregida en {name}: factor {correction_factor:.4f}")

        return model

    def _automatic_synchronization(self, model: nn.Module) -> nn.Module:
        """
        Aplica sincronización automática.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo sincronizado
        """
        # Sincronizar todos los parámetros
        for param in model.parameters():
            if param.requires_grad:
                # All-reduce para sincronización
                dist.all_reduce(param.data, op=dist.ReduceOp.AVG)

        logger.info("Sincronización automática aplicada")

        return model

    def _consensus_synchronization(self, model: nn.Module) -> nn.Module:
        """
        Aplica sincronización por consensus.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con consensus aplicado
        """
        world_size = dist.get_world_size()

        for iteration in range(self.config.max_sync_iterations):
            consensus_reached = True

            for name, param in model.named_parameters():
                if 'weight' in name:
                    # Calcular consensus
                    param_copy = param.data.clone()
                    dist.all_reduce(param_copy, op=dist.ReduceOp.AVG)

                    # Verificar si se alcanzó consensus
                    diff = torch.norm(param.data - param_copy).item()
                    if diff > self.config.consensus_threshold:
                        consensus_reached = False
                        param.data.copy_(param_copy)

            if consensus_reached:
                break

            self.sync_stats['consensus_iterations'] += 1

        logger.info(f"Consensus alcanzado en {iteration + 1} iteraciones")

        return model

    def calculate_coherence_score(self) -> float:
        """
        Calcula el score de coherencia.

        Returns:
            Score de coherencia (0-1)
        """
        if not self.weight_history:
            return 1.0

        total_coherence = 0.0
        layer_count = 0

        for name, norms in self.weight_history.items():
            if len(norms) >= 3:
                # Calcular estabilidad de normas
                recent_norms = norms[-3:]
                stability = 1.0 / (1.0 + np.std(recent_norms))
                total_coherence += stability
                layer_count += 1

        if layer_count == 0:
            return 1.0

        coherence_score = total_coherence / layer_count
        self.sync_stats['coherence_score'] = coherence_score

        return coherence_score

    def get_synchronization_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de sincronización.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'sync_stats': self.sync_stats.copy(),
            'sync_history': self.sync_history.copy(),
            'coherence_score': self.calculate_coherence_score()
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de sincronización."""
        self.weight_history.clear()
        self.sync_history.clear()
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'divergence_detections': 0,
            'consensus_iterations': 0,
            'sync_latency': 0.0,
            'coherence_score': 0.0
        }

        logger.info("Estadísticas de sincronización reiniciadas")
