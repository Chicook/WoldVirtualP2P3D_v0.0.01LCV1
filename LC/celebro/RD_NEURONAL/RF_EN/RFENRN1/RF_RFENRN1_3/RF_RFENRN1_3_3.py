"""
RF_RFENRN1_3_3.py - Gestor de Regularización Adaptativa
========================================================

Implementa técnicas avanzadas de regularización adaptativa para redes neuronales
de aprendizaje por refuerzo. Incluye L1/L2 adaptativos, Dropout inteligente,
SpectralNorm, WeightDecay dinámico y técnicas de ElasticNet que se ajustan
automáticamente durante el entrenamiento.

Características:
- Regularización L1/L2 adaptativa con coeficientes dinámicos
- Dropout inteligente basado en importancia de neuronas
- SpectralNorm para estabilidad en GANs y redes adversarias
- WeightDecay adaptativo por capas
- ElasticNet con balanceado automático
- DropConnect para regularización de conexiones
- Análisis de importancia de pesos para regularización selectiva

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

logger = logging.getLogger('RFENRN1.RF_RFENRN1_3_3')


@dataclass
class RegularizationConfig:
    """Configuración para técnicas de regularización"""
    use_l1_regularization: bool = True
    use_l2_regularization: bool = True
    use_elastic_net: bool = True
    use_adaptive_dropout: bool = True
    use_spectral_norm: bool = False
    use_weight_decay: bool = True
    use_drop_connect: bool = False
    l1_lambda: float = 0.01
    l2_lambda: float = 0.01
    elastic_alpha: float = 0.5
    dropout_rate: float = 0.1
    weight_decay_rate: float = 1e-4
    spectral_norm_power_iterations: int = 1
    adaptive_threshold: float = 0.1
    importance_threshold: float = 0.05
    regularization_strategy: str = 'adaptive'  # 'fixed', 'adaptive', 'mixed'


class DynamicRegularizationManager:
    """
    Gestor de regularización adaptativa para redes de refuerzo.

    Implementa técnicas avanzadas de regularización que se adaptan
    automáticamente al comportamiento del modelo durante el entrenamiento.
    """

    def __init__(self, config: Optional[RegularizationConfig] = None):
        """
        Inicializa el gestor de regularización.

        Args:
            config: Configuración de regularización (opcional)
        """
        self.config = config or RegularizationConfig()
        self.regularization_layers = {}
        self.weight_importance = {}
        self.regularization_stats = {
            'l1_penalty': 0.0,
            'l2_penalty': 0.0,
            'elastic_penalty': 0.0,
            'dropout_efficiency': 0.0,
            'spectral_norm_value': 0.0,
            'regularization_effectiveness': 0.0
        }
        self.adaptation_history = defaultdict(list)

        logger.info("DynamicRegularizationManager inicializado")

    def apply_regularization_to_model(self, model: nn.Module) -> nn.Module:
        """
        Aplica regularización al modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con regularización aplicada
        """
        if self.config.regularization_strategy == 'adaptive':
            return self._apply_adaptive_regularization(model)
        elif self.config.regularization_strategy == 'mixed':
            return self._apply_mixed_regularization(model)
        else:
            return self._apply_fixed_regularization(model)

    def _apply_adaptive_regularization(self, model: nn.Module) -> nn.Module:
        """
        Aplica regularización adaptativa al modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con regularización adaptativa
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Aplicar diferentes tipos de regularización
                if self.config.use_spectral_norm:
                    module = self._apply_spectral_norm(module, name)

                if self.config.use_adaptive_dropout:
                    module = self._apply_adaptive_dropout(module, name)

                if self.config.use_drop_connect:
                    module = self._apply_drop_connect(module, name)

                self.regularization_layers[name] = module

        logger.info("Regularización adaptativa aplicada al modelo")
        return model

    def _apply_spectral_norm(self, module: nn.Module, name: str) -> nn.Module:
        """
        Aplica SpectralNorm al módulo.

        Args:
            module: Módulo a regularizar
            name: Nombre del módulo

        Returns:
            Módulo con SpectralNorm aplicado
        """
        try:
            if isinstance(module, nn.Linear):
                return nn.utils.spectral_norm(module, n_power_iterations=self.config.spectral_norm_power_iterations)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                return nn.utils.spectral_norm(module, n_power_iterations=self.config.spectral_norm_power_iterations)
        except Exception as e:
            logger.warning(f"Error aplicando SpectralNorm a {name}: {e}")

        return module

    def _apply_adaptive_dropout(self, module: nn.Module, name: str) -> nn.Module:
        """
        Aplica Dropout adaptativo al módulo.

        Args:
            module: Módulo a regularizar
            name: Nombre del módulo

        Returns:
            Módulo con Dropout adaptativo
        """
        # Crear wrapper de dropout adaptativo
        adaptive_dropout = AdaptiveDropout(self.config.dropout_rate, self.config)
        return AdaptiveDropoutWrapper(module, adaptive_dropout)

    def _apply_drop_connect(self, module: nn.Module, name: str) -> nn.Module:
        """
        Aplica DropConnect al módulo.

        Args:
            module: Módulo a regularizar
            name: Nombre del módulo

        Returns:
            Módulo con DropConnect aplicado
        """
        drop_connect = DropConnect(self.config.dropout_rate)
        return DropConnectWrapper(module, drop_connect)

    def calculate_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Calcula la pérdida de regularización.

        Args:
            model: Modelo PyTorch

        Returns:
            Pérdida de regularización total
        """
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if 'weight' in name:
                # L1 Regularization
                if self.config.use_l1_regularization:
                    l1_loss = self._calculate_l1_loss(param)
                    total_loss += self.config.l1_lambda * l1_loss

                # L2 Regularization
                if self.config.use_l2_regularization:
                    l2_loss = self._calculate_l2_loss(param)
                    total_loss += self.config.l2_lambda * l2_loss

                # ElasticNet Regularization
                if self.config.use_elastic_net:
                    elastic_loss = self._calculate_elastic_net_loss(param)
                    total_loss += self.config.elastic_alpha * elastic_loss

        return total_loss

    def _calculate_l1_loss(self, param: torch.Tensor) -> torch.Tensor:
        """Calcula pérdida L1."""
        return torch.sum(torch.abs(param))

    def _calculate_l2_loss(self, param: torch.Tensor) -> torch.Tensor:
        """Calcula pérdida L2."""
        return torch.sum(param ** 2)

    def _calculate_elastic_net_loss(self, param: torch.Tensor) -> torch.Tensor:
        """Calcula pérdida ElasticNet."""
        l1_loss = self._calculate_l1_loss(param)
        l2_loss = self._calculate_l2_loss(param)
        return l1_loss + l2_loss

    def update_regularization_parameters(self, model: nn.Module, epoch: int) -> None:
        """
        Actualiza parámetros de regularización.

        Args:
            model: Modelo PyTorch
            epoch: Época actual
        """
        # Actualizar importancia de pesos
        self._update_weight_importance(model)

        # Actualizar coeficientes de regularización
        self._update_regularization_coefficients(epoch)

        # Actualizar dropout adaptativo
        self._update_adaptive_dropout(model, epoch)

    def _update_weight_importance(self, model: nn.Module) -> None:
        """
        Actualiza importancia de pesos.

        Args:
            model: Modelo PyTorch
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Calcular importancia basándose en magnitud y gradientes
                magnitude = torch.norm(param.data).item()
                if param.grad is not None:
                    gradient_magnitude = torch.norm(param.grad).item()
                    importance = magnitude * gradient_magnitude
                else:
                    importance = magnitude

                self.weight_importance[name] = importance

    def _update_regularization_coefficients(self, epoch: int) -> None:
        """
        Actualiza coeficientes de regularización.

        Args:
            epoch: Época actual
        """
        # Aumentar regularización gradualmente
        progress = min(epoch / 100.0, 1.0)

        # Ajustar L1 lambda
        if self.config.use_l1_regularization:
            self.config.l1_lambda *= (1.0 + progress * 0.1)

        # Ajustar L2 lambda
        if self.config.use_l2_regularization:
            self.config.l2_lambda *= (1.0 + progress * 0.1)

    def _update_adaptive_dropout(self, model: nn.Module, epoch: int) -> None:
        """
        Actualiza Dropout adaptativo.

        Args:
            model: Modelo PyTorch
            epoch: Época actual
        """
        for name, module in model.named_modules():
            if isinstance(module, AdaptiveDropoutWrapper):
                # Ajustar tasa de dropout basándose en importancia
                if name in self.weight_importance:
                    importance = self.weight_importance[name]
                    adaptive_rate = self.config.dropout_rate * (1.0 - importance)
                    module.dropout.update_rate(adaptive_rate)

    def calculate_regularization_effectiveness(self) -> float:
        """
        Calcula la efectividad de la regularización.

        Returns:
            Efectividad de regularización (0-1)
        """
        if not self.weight_importance:
            return 0.0

        # Calcular distribución de importancia
        importances = list(self.weight_importance.values())
        mean_importance = np.mean(importances)
        std_importance = np.std(importances)

        # Efectividad basada en distribución de importancia
        effectiveness = 1.0 / (1.0 + std_importance / mean_importance) if mean_importance > 0 else 0.0

        self.regularization_stats['regularization_effectiveness'] = effectiveness

        return effectiveness

    def get_regularization_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de regularización.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'regularization_stats': self.regularization_stats.copy(),
            'weight_importance': self.weight_importance.copy(),
            'effectiveness': self.calculate_regularization_effectiveness()
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de regularización."""
        self.weight_importance.clear()
        self.adaptation_history.clear()
        self.regularization_stats = {
            'l1_penalty': 0.0,
            'l2_penalty': 0.0,
            'elastic_penalty': 0.0,
            'dropout_efficiency': 0.0,
            'spectral_norm_value': 0.0,
            'regularization_effectiveness': 0.0
        }

        logger.info("Estadísticas de regularización reiniciadas")


class AdaptiveDropout:
    """Dropout adaptativo basado en importancia de neuronas."""

    def __init__(self, base_rate: float, config: RegularizationConfig):
        self.base_rate = base_rate
        self.current_rate = base_rate
        self.config = config
        self.importance_history = []

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass con dropout adaptativo."""
        if not training:
            return x

        # Calcular importancia de cada neurona
        importance = torch.abs(x)
        self.importance_history.append(importance.mean().item())

        if len(self.importance_history) > 10:
            self.importance_history.pop(0)

        # Ajustar tasa de dropout basándose en importancia
        if len(self.importance_history) >= 5:
            avg_importance = np.mean(self.importance_history)
            self.current_rate = self.base_rate * (1.0 - avg_importance)

        # Aplicar dropout
        mask = torch.rand_like(x) > self.current_rate
        return x * mask.float() / (1.0 - self.current_rate)

    def update_rate(self, new_rate: float) -> None:
        """Actualiza la tasa de dropout."""
        self.current_rate = max(0.0, min(1.0, new_rate))


class AdaptiveDropoutWrapper(nn.Module):
    """Wrapper para aplicar dropout adaptativo."""

    def __init__(self, base_module: nn.Module, dropout: AdaptiveDropout):
        super().__init__()
        self.base_module = base_module
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con dropout adaptativo."""
        output = self.base_module(x)
        return self.dropout.forward(output, self.training)


class DropConnect:
    """DropConnect para regularización de conexiones."""

    def __init__(self, drop_rate: float):
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass con DropConnect."""
        if not training:
            return x

        # Aplicar DropConnect
        mask = torch.rand_like(x) > self.drop_rate
        return x * mask.float() / (1.0 - self.drop_rate)


class DropConnectWrapper(nn.Module):
    """Wrapper para aplicar DropConnect."""

    def __init__(self, base_module: nn.Module, drop_connect: DropConnect):
        super().__init__()
        self.base_module = base_module
        self.drop_connect = drop_connect

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con DropConnect."""
        output = self.base_module(x)
        return self.drop_connect.forward(output, self.training)
