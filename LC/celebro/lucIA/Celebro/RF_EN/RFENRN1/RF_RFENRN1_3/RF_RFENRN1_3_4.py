"""
RF_RFENRN1_3_4.py - Ajustador Dinámico de Pesos
===============================================

Implementa técnicas avanzadas de ajuste dinámico de pesos durante el entrenamiento
de redes neuronales de aprendizaje por refuerzo. Incluye escalado adaptativo,
ajuste basado en gradientes, momentum dinámico y técnicas de annealing para
optimizar la convergencia y estabilidad del modelo.

Características:
- Escalado adaptativo de pesos basado en estadísticas
- Ajuste dinámico basado en gradientes y momentum
- Momentum adaptativo con decay inteligente
- Annealing de pesos para convergencia suave
- Ajuste por capas con diferentes estrategias
- Detección automática de problemas de convergencia
- Corrección automática de pesos divergentes

Autor: LucIA Development Team
Versión: 3.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger('RFENRN1.RF_RFENRN1_3_4')


@dataclass
class DynamicAdjustmentConfig:
    """Configuración para ajuste dinámico de pesos"""
    use_adaptive_scaling: bool = True
    use_gradient_based_adjustment: bool = True
    use_momentum_adjustment: bool = True
    use_weight_annealing: bool = True
    use_layer_wise_adjustment: bool = True
    scaling_factor: float = 1.0
    momentum_decay: float = 0.9
    annealing_rate: float = 0.95
    gradient_threshold: float = 1.0
    divergence_threshold: float = 10.0
    adjustment_frequency: int = 10
    convergence_window: int = 50


class DynamicWeightAdjuster:
    """
    Ajustador dinámico de pesos para redes de refuerzo.

    Implementa técnicas avanzadas de ajuste dinámico que se adaptan
    automáticamente al comportamiento del modelo durante el entrenamiento.
    """

    def __init__(self, config: Optional[DynamicAdjustmentConfig] = None):
        """
        Inicializa el ajustador dinámico.

        Args:
            config: Configuración de ajuste dinámico (opcional)
        """
        self.config = config or DynamicAdjustmentConfig()
        self.weight_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.adjustment_stats = {
            'total_adjustments': 0,
            'scaling_adjustments': 0,
            'momentum_adjustments': 0,
            'annealing_adjustments': 0,
            'divergence_corrections': 0,
            'convergence_rate': 0.0
        }
        self.layer_strategies = {}

        logger.info("DynamicWeightAdjuster inicializado")

    def adjust_model_weights(self, model: nn.Module, epoch: int, loss: float) -> nn.Module:
        """
        Ajusta pesos del modelo dinámicamente.

        Args:
            model: Modelo PyTorch
            epoch: Época actual
            loss: Pérdida actual

        Returns:
            Modelo con pesos ajustados
        """
        if epoch % self.config.adjustment_frequency != 0:
            return model

        # Detectar problemas de convergencia
        convergence_issues = self._detect_convergence_issues(model, loss)

        # Aplicar ajustes basándose en problemas detectados
        if convergence_issues['divergence']:
            model = self._correct_divergence(model)
        elif convergence_issues['slow_convergence']:
            model = self._accelerate_convergence(model)
        elif convergence_issues['oscillation']:
            model = self._stabilize_oscillation(model)

        # Ajustes regulares
        if self.config.use_adaptive_scaling:
            model = self._apply_adaptive_scaling(model)

        if self.config.use_momentum_adjustment:
            model = self._apply_momentum_adjustment(model)

        if self.config.use_weight_annealing:
            model = self._apply_weight_annealing(model, epoch)

        self.adjustment_stats['total_adjustments'] += 1

        return model

    def _detect_convergence_issues(self, model: nn.Module, loss: float) -> Dict[str, bool]:
        """
        Detecta problemas de convergencia.

        Args:
            model: Modelo PyTorch
            loss: Pérdida actual

        Returns:
            Diccionario con problemas detectados
        """
        issues = {
            'divergence': False,
            'slow_convergence': False,
            'oscillation': False
        }

        # Detectar divergencia
        if loss > self.config.divergence_threshold:
            issues['divergence'] = True

        # Detectar convergencia lenta
        if len(self.weight_history['loss']) >= self.config.convergence_window:
            recent_losses = self.weight_history['loss'][-self.config.convergence_window:]
            if np.std(recent_losses) < 0.01 and np.mean(recent_losses) > 0.1:
                issues['slow_convergence'] = True

        # Detectar oscilación
        if len(self.weight_history['loss']) >= 10:
            recent_losses = self.weight_history['loss'][-10:]
            if np.std(recent_losses) > np.mean(recent_losses) * 0.5:
                issues['oscillation'] = True

        # Registrar pérdida
        self.weight_history['loss'].append(loss)

        return issues

    def _correct_divergence(self, model: nn.Module) -> nn.Module:
        """
        Corrige pesos divergentes.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con pesos corregidos
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Detectar pesos divergentes
                weight_norm = torch.norm(param.data).item()
                if weight_norm > self.config.divergence_threshold:
                    # Escalar pesos hacia abajo
                    scale_factor = self.config.divergence_threshold / weight_norm
                    param.data *= scale_factor

                    logger.warning(f"Pesos divergentes corregidos en {name}: escala {scale_factor:.4f}")

        self.adjustment_stats['divergence_corrections'] += 1
        return model

    def _accelerate_convergence(self, model: nn.Module) -> nn.Module:
        """
        Acelera la convergencia del modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con convergencia acelerada
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Aumentar magnitud de pesos para acelerar
                weight_norm = torch.norm(param.data).item()
                if weight_norm < 0.1:
                    scale_factor = 1.5
                    param.data *= scale_factor

        return model

    def _stabilize_oscillation(self, model: nn.Module) -> nn.Module:
        """
        Estabiliza oscilaciones en los pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo estabilizado
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Suavizar pesos para reducir oscilación
                if name in self.weight_history:
                    if len(self.weight_history[name]) >= 3:
                        # Promedio móvil de pesos
                        recent_weights = self.weight_history[name][-3:]
                        smoothed_weight = torch.mean(torch.stack(recent_weights), dim=0)
                        param.data = 0.7 * param.data + 0.3 * smoothed_weight

        return model

    def _apply_adaptive_scaling(self, model: nn.Module) -> nn.Module:
        """
        Aplica escalado adaptativo a los pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con escalado adaptativo
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Calcular factor de escalado basándose en estadísticas
                weight_mean = param.data.mean().item()
                weight_std = param.data.std().item()

                # Factor de escalado adaptativo
                if weight_std > 0:
                    scale_factor = self.config.scaling_factor * (1.0 / (1.0 + weight_std))
                    param.data *= scale_factor

                    # Registrar historial
                    self.weight_history[name].append(param.data.clone())
                    if len(self.weight_history[name]) > 10:
                        self.weight_history[name].pop(0)

        self.adjustment_stats['scaling_adjustments'] += 1
        return model

    def _apply_momentum_adjustment(self, model: nn.Module) -> nn.Module:
        """
        Aplica ajuste de momentum a los pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con momentum ajustado
        """
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                # Calcular momentum adaptativo
                grad_norm = torch.norm(param.grad).item()

                if grad_norm > self.config.gradient_threshold:
                    # Reducir momentum para gradientes grandes
                    momentum_factor = self.config.momentum_decay * (1.0 / (1.0 + grad_norm))
                else:
                    # Aumentar momentum para gradientes pequeños
                    momentum_factor = self.config.momentum_decay * (1.0 + grad_norm)

                # Aplicar ajuste de momentum
                param.data += momentum_factor * param.grad

        self.adjustment_stats['momentum_adjustments'] += 1
        return model

    def _apply_weight_annealing(self, model: nn.Module, epoch: int) -> nn.Module:
        """
        Aplica annealing de pesos.

        Args:
            model: Modelo PyTorch
            epoch: Época actual

        Returns:
            Modelo con annealing aplicado
        """
        # Factor de annealing basado en época
        annealing_factor = self.config.annealing_rate ** (epoch / 100.0)

        for name, param in model.named_parameters():
            if 'weight' in name:
                # Aplicar annealing gradual
                param.data *= annealing_factor

        self.adjustment_stats['annealing_adjustments'] += 1
        return model

    def calculate_convergence_rate(self) -> float:
        """
        Calcula la tasa de convergencia.

        Returns:
            Tasa de convergencia
        """
        if len(self.weight_history['loss']) < 10:
            return 0.0

        recent_losses = self.weight_history['loss'][-10:]

        # Calcular tasa de mejora
        if len(recent_losses) >= 2:
            improvement_rate = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
            convergence_rate = max(0.0, improvement_rate)
        else:
            convergence_rate = 0.0

        self.adjustment_stats['convergence_rate'] = convergence_rate

        return convergence_rate

    def get_adjustment_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de ajuste.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'adjustment_stats': self.adjustment_stats.copy(),
            'convergence_rate': self.calculate_convergence_rate(),
            'weight_history_size': len(self.weight_history)
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de ajuste."""
        self.weight_history.clear()
        self.gradient_history.clear()
        self.adjustment_stats = {
            'total_adjustments': 0,
            'scaling_adjustments': 0,
            'momentum_adjustments': 0,
            'annealing_adjustments': 0,
            'divergence_corrections': 0,
            'convergence_rate': 0.0
        }

        logger.info("Estadísticas de ajuste dinámico reiniciadas")
