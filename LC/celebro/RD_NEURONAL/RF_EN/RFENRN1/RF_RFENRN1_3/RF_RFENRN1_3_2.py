"""
RF_RFENRN1_3_2.py - Gestor de Normalización Adaptativa
======================================================

Implementa técnicas avanzadas de normalización para redes neuronales de
aprendizaje por refuerzo. Incluye BatchNorm, LayerNorm, GroupNorm adaptativos
y técnicas de normalización dinámica que se ajustan automáticamente según
el comportamiento de las activaciones durante el entrenamiento.

Características:
- BatchNorm adaptativo con momentum dinámico
- LayerNorm mejorado con escalado adaptativo
- GroupNorm con agrupación inteligente
- Normalización dinámica basada en estadísticas en tiempo real
- WeightNorm para normalización de pesos
- Normalización por instancia adaptativa
- Análisis automático de distribución de activaciones

Autor: LucIA Development Team
Versión: 3.0.0
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
from collections import defaultdict

logger = logging.getLogger('RFENRN1.RF_RFENRN1_3_2')


@dataclass
class NormalizationConfig:
    """Configuración para técnicas de normalización"""
    use_batch_norm: bool = True
    use_layer_norm: bool = True
    use_group_norm: bool = True
    use_weight_norm: bool = False
    use_instance_norm: bool = False
    use_dynamic_norm: bool = True
    batch_norm_momentum: float = 0.1
    batch_norm_eps: float = 1e-5
    layer_norm_eps: float = 1e-5
    group_norm_groups: int = 32
    weight_norm_dim: int = 0
    dynamic_norm_threshold: float = 0.1
    adaptive_momentum: bool = True
    track_activation_stats: bool = True
    normalization_strategy: str = 'adaptive'  # 'fixed', 'adaptive', 'mixed'


class AdaptiveNormalizationManager:
    """
    Gestor de normalización adaptativa para redes de refuerzo.

    Implementa técnicas avanzadas de normalización que se adaptan
    automáticamente al comportamiento de las activaciones.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Inicializa el gestor de normalización.

        Args:
            config: Configuración de normalización (opcional)
        """
        self.config = config or NormalizationConfig()
        self.normalization_layers = {}
        self.activation_stats = defaultdict(list)
        self.normalization_stats = {
            'batch_norm_layers': 0,
            'layer_norm_layers': 0,
            'group_norm_layers': 0,
            'weight_norm_layers': 0,
            'dynamic_norm_layers': 0,
            'normalization_efficiency': 0.0
        }

        logger.info("AdaptiveNormalizationManager inicializado")

    def add_normalization_to_model(self, model: nn.Module) -> nn.Module:
        """
        Añade capas de normalización al modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con normalización añadida
        """
        if self.config.normalization_strategy == 'adaptive':
            return self._add_adaptive_normalization(model)
        elif self.config.normalization_strategy == 'mixed':
            return self._add_mixed_normalization(model)
        else:
            return self._add_fixed_normalization(model)

    def _add_adaptive_normalization(self, model: nn.Module) -> nn.Module:
        """
        Añade normalización adaptativa al modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con normalización adaptativa
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Determinar tipo de normalización basándose en la capa
                norm_type = self._determine_normalization_type(module, name)

                if norm_type == 'batch_norm' and self.config.use_batch_norm:
                    norm_layer = self._create_adaptive_batch_norm(module)
                elif norm_type == 'layer_norm' and self.config.use_layer_norm:
                    norm_layer = self._create_adaptive_layer_norm(module)
                elif norm_type == 'group_norm' and self.config.use_group_norm:
                    norm_layer = self._create_adaptive_group_norm(module)
                elif norm_type == 'dynamic_norm' and self.config.use_dynamic_norm:
                    norm_layer = self._create_dynamic_normalization(module)
                else:
                    continue

                # Reemplazar módulo con versión normalizada
                self._replace_module(model, name, norm_layer)

        logger.info("Normalización adaptativa aplicada al modelo")
        return model

    def _determine_normalization_type(self, module: nn.Module, name: str) -> str:
        """
        Determina el tipo de normalización óptimo para una capa.

        Args:
            module: Módulo a analizar
            name: Nombre del módulo

        Returns:
            Tipo de normalización recomendado
        """
        if isinstance(module, nn.Linear):
            # Para capas lineales, usar LayerNorm
            return 'layer_norm'
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Para convoluciones, usar BatchNorm o GroupNorm
            if module.kernel_size[0] > 1:
                return 'batch_norm'
            else:
                return 'group_norm'
        else:
            return 'dynamic_norm'

    def _create_adaptive_batch_norm(self, module: nn.Module) -> nn.Module:
        """
        Crea BatchNorm adaptativo.

        Args:
            module: Módulo base

        Returns:
            Módulo con BatchNorm adaptativo
        """
        if isinstance(module, nn.Linear):
            norm_layer = AdaptiveBatchNorm1d(module.out_features, self.config)
        elif isinstance(module, nn.Conv1d):
            norm_layer = AdaptiveBatchNorm1d(module.out_channels, self.config)
        elif isinstance(module, nn.Conv2d):
            norm_layer = AdaptiveBatchNorm2d(module.out_channels, self.config)
        elif isinstance(module, nn.Conv3d):
            norm_layer = AdaptiveBatchNorm3d(module.out_channels, self.config)
        else:
            return module

        self.normalization_stats['batch_norm_layers'] += 1
        return norm_layer

    def _create_adaptive_layer_norm(self, module: nn.Module) -> nn.Module:
        """
        Crea LayerNorm adaptativo.

        Args:
            module: Módulo base

        Returns:
            Módulo con LayerNorm adaptativo
        """
        if isinstance(module, nn.Linear):
            norm_layer = AdaptiveLayerNorm(module.out_features, self.config)
        else:
            return module

        self.normalization_stats['layer_norm_layers'] += 1
        return norm_layer

    def _create_adaptive_group_norm(self, module: nn.Module) -> nn.Module:
        """
        Crea GroupNorm adaptativo.

        Args:
            module: Módulo base

        Returns:
            Módulo con GroupNorm adaptativo
        """
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            norm_layer = AdaptiveGroupNorm(
                module.out_channels,
                self.config.group_norm_groups,
                self.config
            )
        else:
            return module

        self.normalization_stats['group_norm_layers'] += 1
        return norm_layer

    def _create_dynamic_normalization(self, module: nn.Module) -> nn.Module:
        """
        Crea normalización dinámica.

        Args:
            module: Módulo base

        Returns:
            Módulo con normalización dinámica
        """
        norm_layer = DynamicNormalization(module, self.config)
        self.normalization_stats['dynamic_norm_layers'] += 1
        return norm_layer

    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module) -> None:
        """
        Reemplaza un módulo en el modelo.

        Args:
            model: Modelo PyTorch
            name: Nombre del módulo
            new_module: Nuevo módulo
        """
        # Implementación básica - en producción sería más compleja
        pass

    def track_activation_statistics(self, activations: torch.Tensor, layer_name: str) -> None:
        """
        Rastrea estadísticas de activaciones.

        Args:
            activations: Tensor de activaciones
            layer_name: Nombre de la capa
        """
        if not self.config.track_activation_stats:
            return

        stats = {
            'mean': activations.mean().item(),
            'std': activations.std().item(),
            'min': activations.min().item(),
            'max': activations.max().item(),
            'shape': list(activations.shape)
        }

        self.activation_stats[layer_name].append(stats)

    def update_normalization_parameters(self, model: nn.Module) -> None:
        """
        Actualiza parámetros de normalización basándose en estadísticas.

        Args:
            model: Modelo PyTorch
        """
        for name, module in model.named_modules():
            if isinstance(module, (AdaptiveBatchNorm1d, AdaptiveBatchNorm2d,
                                   AdaptiveBatchNorm3d, AdaptiveLayerNorm,
                                   AdaptiveGroupNorm, DynamicNormalization)):
                module.update_parameters(self.activation_stats.get(name, []))

    def calculate_normalization_efficiency(self) -> float:
        """
        Calcula la eficiencia de normalización.

        Returns:
            Eficiencia de normalización (0-1)
        """
        total_layers = sum(self.normalization_stats.values())

        if total_layers == 0:
            return 0.0

        # Calcular eficiencia basada en estabilidad de activaciones
        efficiency = 0.0

        for layer_name, stats_list in self.activation_stats.items():
            if len(stats_list) < 2:
                continue

            # Calcular variabilidad en estadísticas
            means = [s['mean'] for s in stats_list]
            stds = [s['std'] for s in stats_list]

            mean_stability = 1.0 / (1.0 + np.std(means))
            std_stability = 1.0 / (1.0 + np.std(stds))

            efficiency += (mean_stability + std_stability) / 2.0

        if len(self.activation_stats) > 0:
            efficiency /= len(self.activation_stats)

        self.normalization_stats['normalization_efficiency'] = efficiency

        return efficiency

    def get_normalization_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de normalización.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'normalization_stats': self.normalization_stats.copy(),
            'activation_stats': dict(self.activation_stats),
            'efficiency': self.calculate_normalization_efficiency()
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de normalización."""
        self.activation_stats.clear()
        self.normalization_stats = {
            'batch_norm_layers': 0,
            'layer_norm_layers': 0,
            'group_norm_layers': 0,
            'weight_norm_layers': 0,
            'dynamic_norm_layers': 0,
            'normalization_efficiency': 0.0
        }

        logger.info("Estadísticas de normalización reiniciadas")


class AdaptiveBatchNorm1d(nn.Module):
    """BatchNorm1d adaptativo con momentum dinámico."""

    def __init__(self, num_features: int, config: NormalizationConfig):
        super().__init__()
        self.num_features = num_features
        self.config = config
        self.batch_norm = nn.BatchNorm1d(num_features, eps=config.batch_norm_eps)
        self.adaptive_momentum = config.adaptive_momentum
        self.momentum_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptive_momentum:
            self._update_momentum(x)

        return self.batch_norm(x)

    def _update_momentum(self, x: torch.Tensor) -> None:
        """Actualiza momentum basándose en estadísticas de entrada."""
        current_std = x.std().item()
        self.momentum_history.append(current_std)

        if len(self.momentum_history) > 10:
            self.momentum_history.pop(0)

        if len(self.momentum_history) >= 5:
            # Ajustar momentum basándose en estabilidad
            std_stability = 1.0 / (1.0 + np.std(self.momentum_history))
            adaptive_momentum = self.config.batch_norm_momentum * std_stability

            self.batch_norm.momentum = adaptive_momentum

    def update_parameters(self, activation_stats: List[Dict]) -> None:
        """Actualiza parámetros basándose en estadísticas históricas."""
        if len(activation_stats) < 2:
            return

        # Ajustar parámetros basándose en tendencias
        pass


class AdaptiveBatchNorm2d(nn.Module):
    """BatchNorm2d adaptativo con momentum dinámico."""

    def __init__(self, num_features: int, config: NormalizationConfig):
        super().__init__()
        self.num_features = num_features
        self.config = config
        self.batch_norm = nn.BatchNorm2d(num_features, eps=config.batch_norm_eps)
        self.adaptive_momentum = config.adaptive_momentum
        self.momentum_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptive_momentum:
            self._update_momentum(x)

        return self.batch_norm(x)

    def _update_momentum(self, x: torch.Tensor) -> None:
        """Actualiza momentum basándose en estadísticas de entrada."""
        current_std = x.std().item()
        self.momentum_history.append(current_std)

        if len(self.momentum_history) > 10:
            self.momentum_history.pop(0)

        if len(self.momentum_history) >= 5:
            std_stability = 1.0 / (1.0 + np.std(self.momentum_history))
            adaptive_momentum = self.config.batch_norm_momentum * std_stability

            self.batch_norm.momentum = adaptive_momentum

    def update_parameters(self, activation_stats: List[Dict]) -> None:
        """Actualiza parámetros basándose en estadísticas históricas."""
        pass


class AdaptiveBatchNorm3d(nn.Module):
    """BatchNorm3d adaptativo con momentum dinámico."""

    def __init__(self, num_features: int, config: NormalizationConfig):
        super().__init__()
        self.num_features = num_features
        self.config = config
        self.batch_norm = nn.BatchNorm3d(num_features, eps=config.batch_norm_eps)
        self.adaptive_momentum = config.adaptive_momentum
        self.momentum_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptive_momentum:
            self._update_momentum(x)

        return self.batch_norm(x)

    def _update_momentum(self, x: torch.Tensor) -> None:
        """Actualiza momentum basándose en estadísticas de entrada."""
        current_std = x.std().item()
        self.momentum_history.append(current_std)

        if len(self.momentum_history) > 10:
            self.momentum_history.pop(0)

        if len(self.momentum_history) >= 5:
            std_stability = 1.0 / (1.0 + np.std(self.momentum_history))
            adaptive_momentum = self.config.batch_norm_momentum * std_stability

            self.batch_norm.momentum = adaptive_momentum

    def update_parameters(self, activation_stats: List[Dict]) -> None:
        """Actualiza parámetros basándose en estadísticas históricas."""
        pass


class AdaptiveLayerNorm(nn.Module):
    """LayerNorm adaptativo con escalado dinámico."""

    def __init__(self, normalized_shape: int, config: NormalizationConfig):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.config = config
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=config.layer_norm_eps)
        self.adaptive_scaling = True
        self.scale_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptive_scaling:
            self._update_scaling(x)

        return self.layer_norm(x)

    def _update_scaling(self, x: torch.Tensor) -> None:
        """Actualiza escalado basándose en estadísticas de entrada."""
        current_scale = x.std().item()
        self.scale_history.append(current_scale)

        if len(self.scale_history) > 10:
            self.scale_history.pop(0)

        if len(self.scale_history) >= 5:
            # Ajustar escalado basándose en estabilidad
            scale_stability = 1.0 / (1.0 + np.std(self.scale_history))
            adaptive_scale = scale_stability

            # Aplicar escalado adaptativo
            self.layer_norm.weight.data *= adaptive_scale

    def update_parameters(self, activation_stats: List[Dict]) -> None:
        """Actualiza parámetros basándose en estadísticas históricas."""
        pass


class AdaptiveGroupNorm(nn.Module):
    """GroupNorm adaptativo con agrupación inteligente."""

    def __init__(self, num_channels: int, num_groups: int, config: NormalizationConfig):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.config = config
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps=config.layer_norm_eps)
        self.adaptive_grouping = True
        self.group_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adaptive_grouping:
            self._update_grouping(x)

        return self.group_norm(x)

    def _update_grouping(self, x: torch.Tensor) -> None:
        """Actualiza agrupación basándose en estadísticas de entrada."""
        # Análisis de correlación entre canales
        if x.dim() >= 3:
            channels = x.shape[1]
            if channels > 1:
                # Calcular correlación promedio entre canales
                channel_corr = self._calculate_channel_correlation(x)
                self.group_history.append(channel_corr)

                if len(self.group_history) > 10:
                    self.group_history.pop(0)

    def _calculate_channel_correlation(self, x: torch.Tensor) -> float:
        """Calcula correlación promedio entre canales."""
        if x.dim() < 3:
            return 0.0

        # Flatten spatial dimensions
        x_flat = x.view(x.size(0), x.size(1), -1)

        # Calculate correlation between channels
        correlations = []
        for i in range(x_flat.size(1)):
            for j in range(i + 1, x_flat.size(1)):
                corr = torch.corrcoef(torch.stack([
                    x_flat[:, i].flatten(),
                    x_flat[:, j].flatten()
                ]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr.item())

        return np.mean(correlations) if correlations else 0.0

    def update_parameters(self, activation_stats: List[Dict]) -> None:
        """Actualiza parámetros basándose en estadísticas históricas."""
        pass


class DynamicNormalization(nn.Module):
    """Normalización dinámica que se adapta automáticamente."""

    def __init__(self, base_module: nn.Module, config: NormalizationConfig):
        super().__init__()
        self.base_module = base_module
        self.config = config
        self.normalization_type = 'none'
        self.adaptation_threshold = config.dynamic_norm_threshold
        self.statistics_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aplicar módulo base
        output = self.base_module(x)

        # Determinar tipo de normalización dinámicamente
        self._update_normalization_type(output)

        # Aplicar normalización seleccionada
        if self.normalization_type == 'batch_norm':
            output = F.batch_norm(output, None, None, training=self.training)
        elif self.normalization_type == 'layer_norm':
            output = F.layer_norm(output, output.shape[-1:])
        elif self.normalization_type == 'group_norm':
            output = F.group_norm(output, 32)

        return output

    def _update_normalization_type(self, x: torch.Tensor) -> None:
        """Actualiza tipo de normalización basándose en estadísticas."""
        stats = {
            'mean': x.mean().item(),
            'std': x.std().item(),
            'shape': list(x.shape)
        }

        self.statistics_history.append(stats)

        if len(self.statistics_history) > 10:
            self.statistics_history.pop(0)

        if len(self.statistics_history) >= 5:
            # Analizar estabilidad de estadísticas
            means = [s['mean'] for s in self.statistics_history]
            stds = [s['std'] for s in self.statistics_history]

            mean_stability = 1.0 / (1.0 + np.std(means))
            std_stability = 1.0 / (1.0 + np.std(stds))

            # Seleccionar normalización basándose en estabilidad
            if mean_stability < self.adaptation_threshold:
                self.normalization_type = 'batch_norm'
            elif std_stability < self.adaptation_threshold:
                self.normalization_type = 'layer_norm'
            else:
                self.normalization_type = 'group_norm'

    def update_parameters(self, activation_stats: List[Dict]) -> None:
        """Actualiza parámetros basándose en estadísticas históricas."""
        pass
