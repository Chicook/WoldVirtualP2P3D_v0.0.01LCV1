"""
RF_RFENRN1_2_3.py - Gestor de Control de Gradientes y Estabilidad
================================================================

Implementa técnicas avanzadas para el control de gradientes y estabilidad
en el entrenamiento de redes neuronales de aprendizaje por refuerzo.
Incluye Gradient Clipping, Gradient Noise, Gradient Centralization y
técnicas de estabilidad numérica.

Características:
- Gradient Clipping (norma y valor) para evitar explosiones
- Gradient Noise para mejorar convergencia y evitar mínimos locales
- Gradient Centralization para estabilidad mejorada
- Detección automática de gradientes problemáticos
- Control adaptativo de la magnitud de gradientes

Autor: LucIA Development Team
Versión: 2.0.0
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
import math
from dataclasses import dataclass
import random

logger = logging.getLogger('RFENRN1.RF_RFENRN1_2_3')


@dataclass
class GradientControlConfig:
    """Configuración para control de gradientes"""
    use_gradient_clipping: bool = True
    clip_norm: float = 1.0
    clip_value: Optional[float] = None
    use_gradient_noise: bool = True
    noise_scale: float = 0.01
    noise_decay: float = 0.99
    use_gradient_centralization: bool = True
    adaptive_clipping: bool = True
    clipping_percentile: float = 95.0
    stability_threshold: float = 10.0
    gradient_norm_history_size: int = 100


class GradientControlManager:
    """
    Gestor de control de gradientes y estabilidad.

    Implementa técnicas avanzadas para mantener gradientes estables
    y evitar problemas de entrenamiento en redes de refuerzo.
    """

    def __init__(self, config: Optional[GradientControlConfig] = None):
        """
        Inicializa el gestor de control de gradientes.

        Args:
            config: Configuración de control de gradientes (opcional)
        """
        self.config = config or GradientControlConfig()
        self.gradient_norm_history = []
        self.noise_scale_current = self.config.noise_scale
        self.stability_metrics = {
            'gradient_norm_mean': 0.0,
            'gradient_norm_std': 0.0,
            'gradient_explosion_count': 0,
            'gradient_vanishing_count': 0,
            'stability_score': 1.0
        }

        logger.info("GradientControlManager inicializado")

    def apply_gradient_clipping(self, model: nn.Module, clip_type: str = 'norm') -> Dict[str, float]:
        """
        Aplica Gradient Clipping al modelo.

        Args:
            model: Modelo PyTorch
            clip_type: Tipo de clipping ('norm' o 'value')

        Returns:
            Métricas de clipping aplicado
        """
        if not self.config.use_gradient_clipping:
            return {'clipped': False, 'original_norm': 0.0, 'clipped_norm': 0.0}

        metrics = {'clipped': False, 'original_norm': 0.0, 'clipped_norm': 0.0}

        # Calcular norma original
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** (1. / 2)
        metrics['original_norm'] = total_norm

        # Aplicar clipping
        if clip_type == 'norm':
            if self.config.adaptive_clipping:
                # Clipping adaptativo basado en percentil histórico
                if len(self.gradient_norm_history) > 10:
                    threshold = np.percentile(self.gradient_norm_history, self.config.clipping_percentile)
                    clip_norm = min(self.config.clip_norm, threshold)
                else:
                    clip_norm = self.config.clip_norm
            else:
                clip_norm = self.config.clip_norm

            clipped_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_norm
            )
            metrics['clipped_norm'] = clipped_norm.item()
            metrics['clipped'] = clipped_norm.item() > clip_norm

        elif clip_type == 'value' and self.config.clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                self.config.clip_value
            )
            metrics['clipped'] = True

        # Actualizar historial
        self.gradient_norm_history.append(total_norm)
        if len(self.gradient_norm_history) > self.config.gradient_norm_history_size:
            self.gradient_norm_history.pop(0)

        return metrics

    def apply_gradient_noise(self, model: nn.Module, step: int) -> None:
        """
        Aplica Gradient Noise para mejorar convergencia.

        Args:
            model: Modelo PyTorch
            step: Paso actual de entrenamiento
        """
        if not self.config.use_gradient_noise:
            return

        # Decay del ruido
        self.noise_scale_current = self.config.noise_scale * (self.config.noise_decay ** step)

        for param in model.parameters():
            if param.grad is not None:
                # Generar ruido gaussiano
                noise = torch.randn_like(param.grad) * self.noise_scale_current
                param.grad.add_(noise)

    def apply_gradient_centralization(self, model: nn.Module) -> None:
        """
        Aplica Gradient Centralization para mejorar estabilidad.

        Args:
            model: Modelo PyTorch
        """
        if not self.config.use_gradient_centralization:
            return

        for param in model.parameters():
            if param.grad is not None and param.dim() > 1:
                # Centralizar gradientes por columna
                grad_mean = param.grad.mean(dim=tuple(range(1, param.dim())), keepdim=True)
                param.grad.sub_(grad_mean)

    def detect_gradient_problems(self, model: nn.Module) -> Dict[str, Any]:
        """
        Detecta problemas potenciales con los gradientes.

        Args:
            model: Modelo PyTorch

        Returns:
            Diccionario con problemas detectados
        """
        problems = {
            'explosion': False,
            'vanishing': False,
            'nan_gradients': False,
            'inf_gradients': False,
            'zero_gradients': False
        }

        total_norm = 0.0
        param_count = 0
        nan_count = 0
        inf_count = 0
        zero_count = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

                # Detectar NaN
                if torch.isnan(param.grad).any():
                    nan_count += 1
                    problems['nan_gradients'] = True

                # Detectar Inf
                if torch.isinf(param.grad).any():
                    inf_count += 1
                    problems['inf_gradients'] = True

                # Detectar gradientes cero
                if param_norm.item() < 1e-8:
                    zero_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1. / 2)

            # Detectar explosión de gradientes
            if total_norm > self.config.stability_threshold:
                problems['explosion'] = True
                self.stability_metrics['gradient_explosion_count'] += 1

            # Detectar gradientes que desaparecen
            if total_norm < 1e-6:
                problems['vanishing'] = True
                self.stability_metrics['gradient_vanishing_count'] += 1

            # Detectar demasiados gradientes cero
            if zero_count / param_count > 0.5:
                problems['zero_gradients'] = True

        return problems

    def calculate_stability_score(self) -> float:
        """
        Calcula un score de estabilidad basado en métricas históricas.

        Returns:
            Score de estabilidad (0-1, mayor es mejor)
        """
        if len(self.gradient_norm_history) < 10:
            return 1.0

        # Calcular estadísticas
        mean_norm = np.mean(self.gradient_norm_history)
        std_norm = np.std(self.gradient_norm_history)

        self.stability_metrics['gradient_norm_mean'] = mean_norm
        self.stability_metrics['gradient_norm_std'] = std_norm

        # Score basado en variabilidad y explosiones
        variability_score = 1.0 / (1.0 + std_norm / mean_norm) if mean_norm > 0 else 0.0
        explosion_penalty = 1.0 / (1.0 + self.stability_metrics['gradient_explosion_count'] * 0.1)
        vanishing_penalty = 1.0 / (1.0 + self.stability_metrics['gradient_vanishing_count'] * 0.1)

        stability_score = variability_score * explosion_penalty * vanishing_penalty
        self.stability_metrics['stability_score'] = stability_score

        return stability_score

    def apply_adaptive_learning_rate(self, optimizer, stability_score: float) -> None:
        """
        Ajusta la tasa de aprendizaje basándose en la estabilidad.

        Args:
            optimizer: Optimizador PyTorch
            stability_score: Score de estabilidad actual
        """
        # Reducir LR si hay problemas de estabilidad
        if stability_score < 0.5:
            for group in optimizer.param_groups:
                group['lr'] *= 0.9
                logger.warning(f"LR reducido a {group['lr']} debido a baja estabilidad")

        # Aumentar LR si la estabilidad es muy alta
        elif stability_score > 0.9:
            for group in optimizer.param_groups:
                group['lr'] *= 1.01
                logger.info(f"LR aumentado a {group['lr']} debido a alta estabilidad")

    def get_stability_metrics(self) -> Dict[str, float]:
        """
        Obtiene las métricas de estabilidad actuales.

        Returns:
            Diccionario con métricas de estabilidad
        """
        return self.stability_metrics.copy()

    def reset_metrics(self) -> None:
        """Reinicia las métricas de estabilidad."""
        self.gradient_norm_history.clear()
        self.noise_scale_current = self.config.noise_scale
        self.stability_metrics = {
            'gradient_norm_mean': 0.0,
            'gradient_norm_std': 0.0,
            'gradient_explosion_count': 0,
            'gradient_vanishing_count': 0,
            'stability_score': 1.0
        }
        logger.info("Métricas de estabilidad reiniciadas")

    def save_state(self, path: str) -> None:
        """
        Guarda el estado del gestor de gradientes.

        Args:
            path: Ruta donde guardar
        """
        state = {
            'config': self.config,
            'gradient_norm_history': self.gradient_norm_history,
            'noise_scale_current': self.noise_scale_current,
            'stability_metrics': self.stability_metrics
        }
        torch.save(state, path)
        logger.info(f"Estado del GradientControlManager guardado en {path}")

    def load_state(self, path: str) -> None:
        """
        Carga el estado del gestor de gradientes.

        Args:
            path: Ruta desde donde cargar
        """
        state = torch.load(path)
        self.config = state.get('config', self.config)
        self.gradient_norm_history = state.get('gradient_norm_history', [])
        self.noise_scale_current = state.get('noise_scale_current', self.config.noise_scale)
        self.stability_metrics = state.get('stability_metrics', self.stability_metrics)
        logger.info(f"Estado del GradientControlManager cargado desde {path}")
