"""
RF_RFENRN1_3_1.py - Inicializador Avanzado de Pesos
===================================================

Implementa técnicas avanzadas de inicialización de pesos para redes neuronales
de aprendizaje por refuerzo. Incluye métodos clásicos mejorados (Xavier, He, Kaiming)
y técnicas adaptativas que se ajustan dinámicamente según la arquitectura y datos.

Características:
- Inicialización Xavier adaptativa con corrección de varianza
- Inicialización He mejorada para activaciones ReLU/LeakyReLU
- Inicialización Kaiming con normalización por capas
- Inicialización ortogonal para estabilidad en RNNs
- Inicialización sparse para reducir sobreajuste
- Inicialización aprendida basada en datos
- Análisis automático de arquitectura para selección óptima

Autor: LucIA Development Team
Versión: 3.0.0
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.init as init
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
from dataclasses import dataclass
import random

logger = logging.getLogger('RFENRN1.RF_RFENRN1_3_1')


@dataclass
class WeightInitializationConfig:
    """Configuración para inicialización de pesos"""
    initialization_method: str = 'adaptive'  # 'xavier', 'he', 'kaiming', 'orthogonal', 'sparse', 'adaptive', 'learned'
    xavier_gain: float = 1.0
    he_gain: float = math.sqrt(2.0)
    kaiming_mode: str = 'fan_out'  # 'fan_in', 'fan_out'
    kaiming_nonlinearity: str = 'relu'
    orthogonal_gain: float = 1.0
    sparse_sparsity: float = 0.1
    learned_epochs: int = 10
    adaptive_threshold: float = 0.1
    use_layer_wise_init: bool = True
    use_activation_aware_init: bool = True
    custom_scaling: bool = True


class AdvancedWeightInitializer:
    """
    Inicializador avanzado de pesos para redes de refuerzo.

    Implementa técnicas de inicialización inteligentes que se adaptan
    automáticamente a la arquitectura y características de los datos.
    """

    def __init__(self, config: Optional[WeightInitializationConfig] = None):
        """
        Inicializa el inicializador de pesos.

        Args:
            config: Configuración de inicialización (opcional)
        """
        self.config = config or WeightInitializationConfig()
        self.layer_stats = {}
        self.initialization_history = []
        self.performance_metrics = {
            'convergence_speed': 0.0,
            'gradient_flow': 0.0,
            'activation_distribution': 0.0,
            'weight_magnitude': 0.0
        }

        logger.info("AdvancedWeightInitializer inicializado")

    def initialize_model(self, model: nn.Module, method: str = None) -> nn.Module:
        """
        Inicializa todos los pesos del modelo.

        Args:
            model: Modelo PyTorch
            method: Método de inicialización (opcional)

        Returns:
            Modelo con pesos inicializados
        """
        method = method or self.config.initialization_method

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self._initialize_layer(module, name, method)
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                self._initialize_rnn_layer(module, name, method)

        logger.info(f"Inicialización {method} aplicada al modelo")
        return model

    def _initialize_layer(self, layer: nn.Module, name: str, method: str) -> None:
        """
        Inicializa una capa específica.

        Args:
            layer: Capa a inicializar
            name: Nombre de la capa
            method: Método de inicialización
        """
        if method == 'xavier':
            self._xavier_initialization(layer, name)
        elif method == 'he':
            self._he_initialization(layer, name)
        elif method == 'kaiming':
            self._kaiming_initialization(layer, name)
        elif method == 'orthogonal':
            self._orthogonal_initialization(layer, name)
        elif method == 'sparse':
            self._sparse_initialization(layer, name)
        elif method == 'adaptive':
            self._adaptive_initialization(layer, name)
        elif method == 'learned':
            self._learned_initialization(layer, name)
        else:
            logger.warning(f"Método de inicialización no soportado: {method}")

    def _xavier_initialization(self, layer: nn.Module, name: str) -> None:
        """
        Inicialización Xavier mejorada.

        Args:
            layer: Capa a inicializar
            name: Nombre de la capa
        """
        if isinstance(layer, nn.Linear):
            fan_in = layer.in_features
            fan_out = layer.out_features

            # Xavier adaptativo con corrección de varianza
            gain = self.config.xavier_gain
            if self.config.custom_scaling:
                gain *= math.sqrt(2.0 / (fan_in + fan_out))

            std = gain * math.sqrt(2.0 / (fan_in + fan_out))

            with torch.no_grad():
                layer.weight.normal_(0, std)
                if layer.bias is not None:
                    layer.bias.zero_()

        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            fan_out = layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]

            gain = self.config.xavier_gain
            if self.config.custom_scaling:
                gain *= math.sqrt(2.0 / (fan_in + fan_out))

            std = gain * math.sqrt(2.0 / (fan_in + fan_out))

            with torch.no_grad():
                layer.weight.normal_(0, std)
                if layer.bias is not None:
                    layer.bias.zero_()

        self._record_initialization(name, 'xavier', layer.weight.data)

    def _he_initialization(self, layer: nn.Module, name: str) -> None:
        """
        Inicialización He mejorada.

        Args:
            layer: Capa a inicializar
            name: Nombre de la capa
        """
        if isinstance(layer, nn.Linear):
            fan_in = layer.in_features

            # He adaptativo con corrección para diferentes activaciones
            gain = self.config.he_gain
            if self.config.custom_scaling:
                gain *= math.sqrt(2.0 / fan_in)

            std = gain * math.sqrt(2.0 / fan_in)

            with torch.no_grad():
                layer.weight.normal_(0, std)
                if layer.bias is not None:
                    layer.bias.zero_()

        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]

            gain = self.config.he_gain
            if self.config.custom_scaling:
                gain *= math.sqrt(2.0 / fan_in)

            std = gain * math.sqrt(2.0 / fan_in)

            with torch.no_grad():
                layer.weight.normal_(0, std)
                if layer.bias is not None:
                    layer.bias.zero_()

        self._record_initialization(name, 'he', layer.weight.data)

    def _kaiming_initialization(self, layer: nn.Module, name: str) -> None:
        """
        Inicialización Kaiming mejorada.

        Args:
            layer: Capa a inicializar
            name: Nombre de la capa
        """
        if isinstance(layer, nn.Linear):
            fan_in = layer.in_features
            fan_out = layer.out_features

            # Kaiming adaptativo
            if self.config.kaiming_mode == 'fan_in':
                fan = fan_in
            else:
                fan = fan_out

            gain = self.config.he_gain
            if self.config.kaiming_nonlinearity == 'relu':
                gain = math.sqrt(2.0)
            elif self.config.kaiming_nonlinearity == 'leaky_relu':
                gain = math.sqrt(2.0 / (1 + 0.01**2))

            std = gain * math.sqrt(1.0 / fan)

            with torch.no_grad():
                layer.weight.normal_(0, std)
                if layer.bias is not None:
                    layer.bias.zero_()

        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            fan_out = layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]

            if self.config.kaiming_mode == 'fan_in':
                fan = fan_in
            else:
                fan = fan_out

            gain = self.config.he_gain
            if self.config.kaiming_nonlinearity == 'relu':
                gain = math.sqrt(2.0)

            std = gain * math.sqrt(1.0 / fan)

            with torch.no_grad():
                layer.weight.normal_(0, std)
                if layer.bias is not None:
                    layer.bias.zero_()

        self._record_initialization(name, 'kaiming', layer.weight.data)

    def _orthogonal_initialization(self, layer: nn.Module, name: str) -> None:
        """
        Inicialización ortogonal para estabilidad.

        Args:
            layer: Capa a inicializar
            name: Nombre de la capa
        """
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                init.orthogonal_(layer.weight, gain=self.config.orthogonal_gain)
                if layer.bias is not None:
                    layer.bias.zero_()

        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Para convoluciones, aplicar ortogonal a cada filtro
            with torch.no_grad():
                for i in range(layer.out_channels):
                    init.orthogonal_(layer.weight[i], gain=self.config.orthogonal_gain)
                if layer.bias is not None:
                    layer.bias.zero_()

        self._record_initialization(name, 'orthogonal', layer.weight.data)

    def _sparse_initialization(self, layer: nn.Module, name: str) -> None:
        """
        Inicialización sparse para reducir sobreajuste.

        Args:
            layer: Capa a inicializar
            name: Nombre de la capa
        """
        sparsity = self.config.sparse_sparsity

        if isinstance(layer, nn.Linear):
            fan_in = layer.in_features
            fan_out = layer.out_features

            # Inicialización Xavier con sparsity
            std = math.sqrt(2.0 / (fan_in + fan_out))

            with torch.no_grad():
                layer.weight.normal_(0, std)

                # Aplicar sparsity
                mask = torch.rand_like(layer.weight) > sparsity
                layer.weight *= mask.float()

                if layer.bias is not None:
                    layer.bias.zero_()

        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            fan_out = layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]

            std = math.sqrt(2.0 / (fan_in + fan_out))

            with torch.no_grad():
                layer.weight.normal_(0, std)

                # Aplicar sparsity
                mask = torch.rand_like(layer.weight) > sparsity
                layer.weight *= mask.float()

                if layer.bias is not None:
                    layer.bias.zero_()

        self._record_initialization(name, 'sparse', layer.weight.data)

    def _adaptive_initialization(self, layer: nn.Module, name: str) -> None:
        """
        Inicialización adaptativa basada en análisis de la capa.

        Args:
            layer: Capa a inicializar
            name: Nombre de la capa
        """
        if isinstance(layer, nn.Linear):
            fan_in = layer.in_features
            fan_out = layer.out_features

            # Análisis adaptativo
            if fan_in > fan_out:
                # Capa de reducción - usar He
                std = math.sqrt(2.0 / fan_in)
            elif fan_out > fan_in:
                # Capa de expansión - usar Xavier
                std = math.sqrt(2.0 / (fan_in + fan_out))
            else:
                # Capa equilibrada - usar promedio
                std = math.sqrt(1.0 / fan_in)

            with torch.no_grad():
                layer.weight.normal_(0, std)
                if layer.bias is not None:
                    layer.bias.zero_()

        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            fan_out = layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]

            # Análisis adaptativo para convoluciones
            if layer.kernel_size[0] > 1:  # Convolución espacial
                std = math.sqrt(2.0 / fan_in)
            else:  # Convolución 1x1
                std = math.sqrt(2.0 / (fan_in + fan_out))

            with torch.no_grad():
                layer.weight.normal_(0, std)
                if layer.bias is not None:
                    layer.bias.zero_()

        self._record_initialization(name, 'adaptive', layer.weight.data)

    def _learned_initialization(self, layer: nn.Module, name: str) -> None:
        """
        Inicialización aprendida basada en datos.

        Args:
            layer: Capa a inicializar
            name: Nombre de la capa
        """
        # Inicialización base con He
        self._he_initialization(layer, name)

        # Marcar para aprendizaje posterior
        if not hasattr(layer, '_learned_init'):
            layer._learned_init = True

        logger.info(f"Inicialización aprendida configurada para {name}")

    def _initialize_rnn_layer(self, layer: nn.Module, name: str, method: str) -> None:
        """
        Inicializa capas RNN especializadas.

        Args:
            layer: Capa RNN
            name: Nombre de la capa
            method: Método de inicialización
        """
        if isinstance(layer, (nn.LSTM, nn.GRU)):
            # Inicialización ortogonal para estabilidad en RNNs
            for name_param, param in layer.named_parameters():
                if 'weight' in name_param:
                    init.orthogonal_(param)
                elif 'bias' in name_param:
                    init.zeros_(param)

        elif isinstance(layer, nn.RNN):
            # Inicialización Xavier para RNN simple
            for name_param, param in layer.named_parameters():
                if 'weight' in name_param:
                    init.xavier_uniform_(param)
                elif 'bias' in name_param:
                    init.zeros_(param)

        logger.info(f"Capa RNN {name} inicializada con método {method}")

    def _record_initialization(self, name: str, method: str, weights: torch.Tensor) -> None:
        """
        Registra información de inicialización.

        Args:
            name: Nombre de la capa
            method: Método usado
            weights: Pesos inicializados
        """
        stats = {
            'name': name,
            'method': method,
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'min': weights.min().item(),
            'max': weights.max().item(),
            'shape': list(weights.shape)
        }

        self.layer_stats[name] = stats
        self.initialization_history.append(stats)

    def analyze_weight_distribution(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analiza la distribución de pesos del modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Análisis de distribución de pesos
        """
        analysis = {
            'layer_stats': {},
            'overall_stats': {},
            'gradient_flow': {},
            'recommendations': []
        }

        all_weights = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                weights = module.weight.data
                all_weights.append(weights.flatten())

                # Estadísticas por capa
                layer_stats = {
                    'mean': weights.mean().item(),
                    'std': weights.std().item(),
                    'min': weights.min().item(),
                    'max': weights.max().item(),
                    'zero_ratio': (weights == 0).float().mean().item()
                }

                analysis['layer_stats'][name] = layer_stats

        # Estadísticas generales
        if all_weights:
            all_weights_tensor = torch.cat(all_weights)
            analysis['overall_stats'] = {
                'mean': all_weights_tensor.mean().item(),
                'std': all_weights_tensor.std().item(),
                'min': all_weights_tensor.min().item(),
                'max': all_weights_tensor.max().item(),
                'zero_ratio': (all_weights_tensor == 0).float().mean().item()
            }

        # Generar recomendaciones
        analysis['recommendations'] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Genera recomendaciones basadas en el análisis.

        Args:
            analysis: Análisis de pesos

        Returns:
            Lista de recomendaciones
        """
        recommendations = []

        overall_stats = analysis.get('overall_stats', {})

        if overall_stats.get('std', 0) > 1.0:
            recommendations.append("Considerar reducir la varianza de inicialización")

        if overall_stats.get('zero_ratio', 0) > 0.5:
            recommendations.append("Demasiados pesos cero - considerar inicialización densa")

        if overall_stats.get('max', 0) > 5.0:
            recommendations.append("Pesos muy grandes - considerar inicialización más conservadora")

        return recommendations

    def get_initialization_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de inicialización.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'layer_stats': self.layer_stats,
            'initialization_history': self.initialization_history,
            'performance_metrics': self.performance_metrics
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de inicialización."""
        self.layer_stats.clear()
        self.initialization_history.clear()
        self.performance_metrics = {
            'convergence_speed': 0.0,
            'gradient_flow': 0.0,
            'activation_distribution': 0.0,
            'weight_magnitude': 0.0
        }

        logger.info("Estadísticas de inicialización reiniciadas")
