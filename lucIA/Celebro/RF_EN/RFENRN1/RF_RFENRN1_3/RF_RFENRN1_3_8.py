"""
RF_RFENRN1_3_8.py - Gestor de Recuperación de Pesos
====================================================

Implementa técnicas avanzadas de recuperación de pesos para redes neuronales
de aprendizaje por refuerzo. Incluye recuperación automática de pesos perdidos,
restauración desde checkpoints, recuperación incremental y técnicas de
resiliencia para mantener la integridad del modelo.

Características:
- Recuperación automática de pesos perdidos
- Restauración desde checkpoints múltiples
- Recuperación incremental por capas
- Técnicas de resiliencia y redundancia
- Detección automática de pérdida de pesos
- Recuperación adaptativa basada en contexto
- Validación post-recuperación

Autor: LucIA Development Team
Versión: 3.0.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os
import time
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger('RFENRN1.RF_RFENRN1_3_8')


@dataclass
class WeightRecoveryConfig:
    """Configuración para recuperación de pesos"""
    use_automatic_recovery: bool = True
    use_checkpoint_recovery: bool = True
    use_incremental_recovery: bool = True
    use_resilience_techniques: bool = True
    checkpoint_frequency: int = 100
    max_checkpoints: int = 10
    recovery_timeout: float = 30.0
    validation_after_recovery: bool = True
    backup_frequency: int = 50
    redundancy_factor: float = 0.1


class WeightRecoveryManager:
    """
    Gestor de recuperación de pesos para redes de refuerzo.

    Implementa técnicas avanzadas de recuperación que aseguran
    la integridad y continuidad del entrenamiento.
    """

    def __init__(self, config: Optional[WeightRecoveryConfig] = None):
        """
        Inicializa el gestor de recuperación.

        Args:
            config: Configuración de recuperación (opcional)
        """
        self.config = config or WeightRecoveryConfig()
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'checkpoint_recoveries': 0,
            'incremental_recoveries': 0,
            'automatic_recoveries': 0,
            'recovery_time': 0.0,
            'recovery_success_rate': 0.0
        }
        self.checkpoints = {}
        self.weight_backups = defaultdict(list)
        self.recovery_history = []

        logger.info("WeightRecoveryManager inicializado")

    def save_checkpoint(self, model: nn.Module, step: int, metadata: Dict = None) -> str:
        """
        Guarda un checkpoint del modelo.

        Args:
            model: Modelo PyTorch
            step: Paso actual
            metadata: Metadatos adicionales

        Returns:
            Ruta del checkpoint guardado
        """
        checkpoint_id = f"checkpoint_step_{step}_{int(time.time())}"

        checkpoint_data = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        self.checkpoints[checkpoint_id] = checkpoint_data

        # Mantener solo los últimos N checkpoints
        if len(self.checkpoints) > self.config.max_checkpoints:
            oldest_checkpoint = min(self.checkpoints.keys(),
                                    key=lambda k: self.checkpoints[k]['timestamp'])
            del self.checkpoints[oldest_checkpoint]

        logger.info(f"Checkpoint guardado: {checkpoint_id}")

        return checkpoint_id

    def recover_model_weights(self, model: nn.Module, step: int,
                              recovery_type: str = 'automatic') -> Dict[str, Any]:
        """
        Recupera pesos del modelo.

        Args:
            model: Modelo PyTorch
            step: Paso actual
            recovery_type: Tipo de recuperación

        Returns:
            Resultado de recuperación
        """
        start_time = time.time()

        recovery_result = {
            'step': step,
            'recovery_type': recovery_type,
            'success': False,
            'method_used': None,
            'recovery_time': 0.0,
            'issues': [],
            'metadata': {}
        }

        try:
            if recovery_type == 'automatic':
                recovery_result = self._automatic_recovery(model, step, recovery_result)
            elif recovery_type == 'checkpoint':
                recovery_result = self._checkpoint_recovery(model, step, recovery_result)
            elif recovery_type == 'incremental':
                recovery_result = self._incremental_recovery(model, step, recovery_result)
            elif recovery_type == 'resilience':
                recovery_result = self._resilience_recovery(model, step, recovery_result)

            # Validación post-recuperación
            if self.config.validation_after_recovery and recovery_result['success']:
                validation_result = self._validate_recovery(model)
                recovery_result['validation'] = validation_result

                if not validation_result['passed']:
                    recovery_result['success'] = False
                    recovery_result['issues'].extend(validation_result['issues'])

            # Actualizar estadísticas
            recovery_time = time.time() - start_time
            recovery_result['recovery_time'] = recovery_time

            self.recovery_stats['total_recoveries'] += 1
            if recovery_result['success']:
                self.recovery_stats['successful_recoveries'] += 1
                self.recovery_stats[recovery_type + '_recoveries'] += 1
            else:
                self.recovery_stats['failed_recoveries'] += 1

            self.recovery_stats['recovery_time'] = recovery_time

            # Registrar en historial
            self.recovery_history.append(recovery_result)
            if len(self.recovery_history) > 100:
                self.recovery_history.pop(0)

        except Exception as e:
            logger.error(f"Error en recuperación: {e}")
            recovery_result['success'] = False
            recovery_result['issues'].append(f"Error: {str(e)}")
            recovery_result['recovery_time'] = time.time() - start_time

            self.recovery_stats['failed_recoveries'] += 1

        return recovery_result

    def _automatic_recovery(self, model: nn.Module, step: int,
                            result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recuperación automática.

        Args:
            model: Modelo PyTorch
            step: Paso actual
            result: Resultado de recuperación

        Returns:
            Resultado actualizado
        """
        # Detectar qué pesos necesitan recuperación
        corrupted_layers = self._detect_corrupted_layers(model)

        if not corrupted_layers:
            result['success'] = True
            result['method_used'] = 'no_recovery_needed'
            return result

        # Intentar diferentes métodos de recuperación
        recovery_methods = ['checkpoint', 'incremental', 'resilience']

        for method in recovery_methods:
            try:
                if method == 'checkpoint':
                    result = self._checkpoint_recovery(model, step, result)
                elif method == 'incremental':
                    result = self._incremental_recovery(model, step, result)
                elif method == 'resilience':
                    result = self._resilience_recovery(model, step, result)

                if result['success']:
                    result['method_used'] = method
                    break

            except Exception as e:
                result['issues'].append(f"Error en método {method}: {str(e)}")
                continue

        if not result['success']:
            result['issues'].append("Todos los métodos de recuperación fallaron")

        return result

    def _detect_corrupted_layers(self, model: nn.Module) -> List[str]:
        """
        Detecta capas corruptas.

        Args:
            model: Modelo PyTorch

        Returns:
            Lista de capas corruptas
        """
        corrupted_layers = []

        for name, param in model.named_parameters():
            if 'weight' in name:
                # Verificar corrupción
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    corrupted_layers.append(name)

                # Verificar valores extremos
                weight_norm = torch.norm(param.data).item()
                if weight_norm > 100.0 or weight_norm < 1e-8:
                    corrupted_layers.append(name)

        return corrupted_layers

    def _checkpoint_recovery(self, model: nn.Module, step: int,
                             result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recuperación desde checkpoint.

        Args:
            model: Modelo PyTorch
            step: Paso actual
            result: Resultado de recuperación

        Returns:
            Resultado actualizado
        """
        if not self.checkpoints:
            result['issues'].append("No hay checkpoints disponibles")
            return result

        # Encontrar checkpoint más cercano
        closest_checkpoint = None
        min_distance = float('inf')

        for checkpoint_id, checkpoint_data in self.checkpoints.items():
            checkpoint_step = checkpoint_data['step']
            distance = abs(checkpoint_step - step)

            if distance < min_distance:
                min_distance = distance
                closest_checkpoint = checkpoint_id

        if closest_checkpoint:
            # Restaurar desde checkpoint
            checkpoint_data = self.checkpoints[closest_checkpoint]
            model.load_state_dict(checkpoint_data['model_state_dict'])

            result['success'] = True
            result['method_used'] = 'checkpoint'
            result['metadata']['checkpoint_id'] = closest_checkpoint
            result['metadata']['checkpoint_step'] = checkpoint_data['step']

            logger.info(f"Recuperación desde checkpoint: {closest_checkpoint}")
        else:
            result['issues'].append("No se encontró checkpoint adecuado")

        return result

    def _incremental_recovery(self, model: nn.Module, step: int,
                              result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recuperación incremental por capas.

        Args:
            model: Modelo PyTorch
            step: Paso actual
            result: Resultado de recuperación

        Returns:
            Resultado actualizado
        """
        corrupted_layers = self._detect_corrupted_layers(model)

        if not corrupted_layers:
            result['success'] = True
            result['method_used'] = 'incremental'
            return result

        # Recuperar capas una por una
        recovered_layers = 0

        for layer_name in corrupted_layers:
            try:
                # Buscar backup de esta capa
                if layer_name in self.weight_backups:
                    backups = self.weight_backups[layer_name]
                    if backups:
                        # Usar el backup más reciente
                        latest_backup = backups[-1]

                        # Restaurar peso
                        for name, param in model.named_parameters():
                            if name == layer_name:
                                param.data.copy_(latest_backup)
                                recovered_layers += 1
                                break
                else:
                    # Inicialización de emergencia
                    for name, param in model.named_parameters():
                        if name == layer_name:
                            # Reinicializar con valores pequeños
                            nn.init.normal_(param.data, 0, 0.01)
                            recovered_layers += 1
                            break

            except Exception as e:
                result['issues'].append(f"Error recuperando capa {layer_name}: {str(e)}")

        if recovered_layers > 0:
            result['success'] = True
            result['method_used'] = 'incremental'
            result['metadata']['recovered_layers'] = recovered_layers

            logger.info(f"Recuperación incremental: {recovered_layers} capas")
        else:
            result['issues'].append("No se pudo recuperar ninguna capa")

        return result

    def _resilience_recovery(self, model: nn.Module, step: int,
                             result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recuperación usando técnicas de resiliencia.

        Args:
            model: Modelo PyTorch
            step: Paso actual
            result: Resultado de recuperación

        Returns:
            Resultado actualizado
        """
        # Usar redundancia para reconstruir pesos
        reconstructed_layers = 0

        for name, param in model.named_parameters():
            if 'weight' in name:
                # Verificar si necesita reconstrucción
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    try:
                        # Reconstruir usando información de capas adyacentes
                        reconstructed_weight = self._reconstruct_weight(param, model)
                        if reconstructed_weight is not None:
                            param.data.copy_(reconstructed_weight)
                            reconstructed_layers += 1
                    except Exception as e:
                        result['issues'].append(f"Error reconstruyendo {name}: {str(e)}")

        if reconstructed_layers > 0:
            result['success'] = True
            result['method_used'] = 'resilience'
            result['metadata']['reconstructed_layers'] = reconstructed_layers

            logger.info(f"Recuperación por resiliencia: {reconstructed_layers} capas")
        else:
            result['issues'].append("No se pudo reconstruir ninguna capa")

        return result

    def _reconstruct_weight(self, param: torch.Tensor, model: nn.Module) -> Optional[torch.Tensor]:
        """
        Reconstruye un peso usando información contextual.

        Args:
            param: Parámetro a reconstruir
            model: Modelo completo

        Returns:
            Peso reconstruido o None
        """
        try:
            # Estrategia de reconstrucción basada en estadísticas del modelo
            all_weights = []
            for p in model.parameters():
                if p is not param and torch.isfinite(p.data).all():
                    all_weights.append(p.data.flatten())

            if all_weights:
                all_weights_tensor = torch.cat(all_weights)
                mean_weight = all_weights_tensor.mean()
                std_weight = all_weights_tensor.std()

                # Reconstruir con distribución similar
                reconstructed = torch.randn_like(param.data) * std_weight + mean_weight
                return reconstructed

        except Exception as e:
            logger.error(f"Error en reconstrucción: {e}")

        return None

    def _validate_recovery(self, model: nn.Module) -> Dict[str, Any]:
        """
        Valida la recuperación.

        Args:
            model: Modelo PyTorch

        Returns:
            Resultado de validación
        """
        result = {
            'passed': True,
            'issues': []
        }

        # Verificar integridad básica
        for name, param in model.named_parameters():
            if 'weight' in name:
                if torch.isnan(param.data).any():
                    result['passed'] = False
                    result['issues'].append(f"NaN detectados en {name}")

                if torch.isinf(param.data).any():
                    result['passed'] = False
                    result['issues'].append(f"Inf detectados en {name}")

        return result

    def calculate_recovery_success_rate(self) -> float:
        """
        Calcula la tasa de éxito de recuperación.

        Returns:
            Tasa de éxito (0-1)
        """
        if self.recovery_stats['total_recoveries'] == 0:
            return 0.0

        success_rate = (self.recovery_stats['successful_recoveries'] /
                        self.recovery_stats['total_recoveries'])
        self.recovery_stats['recovery_success_rate'] = success_rate

        return success_rate

    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de recuperación.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'recovery_stats': self.recovery_stats.copy(),
            'recovery_history': self.recovery_history.copy(),
            'success_rate': self.calculate_recovery_success_rate(),
            'checkpoints_available': len(self.checkpoints)
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de recuperación."""
        self.checkpoints.clear()
        self.weight_backups.clear()
        self.recovery_history.clear()
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'checkpoint_recoveries': 0,
            'incremental_recoveries': 0,
            'automatic_recoveries': 0,
            'recovery_time': 0.0,
            'recovery_success_rate': 0.0
        }

        logger.info("Estadísticas de recuperación reiniciadas")
