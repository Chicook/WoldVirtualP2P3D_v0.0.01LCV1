"""
RFEN3_RN_4 - Métodos de Inicialización de Pesos Inteligentes
Implementación de técnicas avanzadas de inicialización para redes neuronales
Incluye: Xavier/Glorot mejorado, He/Kaiming adaptativo, Orthogonal, Sparse, y métodos híbridos
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random

logger = logging.getLogger(__name__)


@dataclass
class WeightInitializationConfig:
    """Configuración para inicialización de pesos"""
    method: str = "xavier_uniform"  # xavier_uniform, xavier_normal, he_uniform, he_normal, orthogonal, sparse
    gain: float = 1.0
    fan_mode: str = "fan_in"  # fan_in, fan_out, fan_avg
    nonlinearity: str = "relu"  # relu, leaky_relu, tanh, sigmoid, linear
    sparse_ratio: float = 0.1
    orthogonal_gain: float = 1.0
    adaptive_init: bool = True
    layer_specific_init: bool = True
    custom_scaling: bool = False


class BaseWeightInitializer(ABC):
    """Clase base abstracta para inicializadores de pesos"""

    def __init__(self, config: WeightInitializationConfig):
        self.config = config

    @abstractmethod
    def initialize(self, tensor: torch.Tensor, layer_type: str = "linear") -> torch.Tensor:
        """Inicializa un tensor de pesos"""
        pass

    def calculate_fan_in_fan_out(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """Calcula fan_in y fan_out para un tensor"""
        if tensor.dim() < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1

        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()

        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out


class XavierUniformInitializer(BaseWeightInitializer):
    """
    Inicializador Xavier/Glorot Uniform mejorado
    """

    def initialize(self, tensor: torch.Tensor, layer_type: str = "linear") -> torch.Tensor:
        """Inicializa con Xavier Uniform"""
        fan_in, fan_out = self.calculate_fan_in_fan_out(tensor)

        # Calcular límites
        if self.config.fan_mode == "fan_in":
            fan = fan_in
        elif self.config.fan_mode == "fan_out":
            fan = fan_out
        else:  # fan_avg
            fan = (fan_in + fan_out) / 2

        # Ajustar gain según la función de activación
        gain = self._get_gain_for_activation()

        # Calcular límite
        limit = gain * math.sqrt(6.0 / fan)

        # Inicializar
        with torch.no_grad():
            tensor.uniform_(-limit, limit)

        return tensor

    def _get_gain_for_activation(self) -> float:
        """Obtiene el gain apropiado para la función de activación"""
        gains = {
            'relu': math.sqrt(2.0),
            'leaky_relu': math.sqrt(2.0 / (1 + 0.01**2)),
            'tanh': 5.0 / 3,
            'sigmoid': 1.0,
            'linear': 1.0
        }
        return gains.get(self.config.nonlinearity, 1.0)


class XavierNormalInitializer(BaseWeightInitializer):
    """
    Inicializador Xavier/Glorot Normal mejorado
    """

    def initialize(self, tensor: torch.Tensor, layer_type: str = "linear") -> torch.Tensor:
        """Inicializa con Xavier Normal"""
        fan_in, fan_out = self.calculate_fan_in_fan_out(tensor)

        # Calcular fan
        if self.config.fan_mode == "fan_in":
            fan = fan_in
        elif self.config.fan_mode == "fan_out":
            fan = fan_out
        else:  # fan_avg
            fan = (fan_in + fan_out) / 2

        # Ajustar gain según la función de activación
        gain = self._get_gain_for_activation()

        # Calcular desviación estándar
        std = gain * math.sqrt(2.0 / fan)

        # Inicializar
        with torch.no_grad():
            tensor.normal_(0, std)

        return tensor

    def _get_gain_for_activation(self) -> float:
        """Obtiene el gain apropiado para la función de activación"""
        gains = {
            'relu': math.sqrt(2.0),
            'leaky_relu': math.sqrt(2.0 / (1 + 0.01**2)),
            'tanh': 5.0 / 3,
            'sigmoid': 1.0,
            'linear': 1.0
        }
        return gains.get(self.config.nonlinearity, 1.0)


class HeUniformInitializer(BaseWeightInitializer):
    """
    Inicializador He/Kaiming Uniform adaptativo
    """

    def initialize(self, tensor: torch.Tensor, layer_type: str = "linear") -> torch.Tensor:
        """Inicializa con He Uniform"""
        fan_in, fan_out = self.calculate_fan_in_fan_out(tensor)

        # Calcular fan según el modo
        if self.config.fan_mode == "fan_in":
            fan = fan_in
        elif self.config.fan_mode == "fan_out":
            fan = fan_out
        else:  # fan_avg
            fan = (fan_in + fan_out) / 2

        # Calcular límite
        limit = math.sqrt(6.0 / fan)

        # Inicializar
        with torch.no_grad():
            tensor.uniform_(-limit, limit)

        return tensor


class HeNormalInitializer(BaseWeightInitializer):
    """
    Inicializador He/Kaiming Normal adaptativo
    """

    def initialize(self, tensor: torch.Tensor, layer_type: str = "linear") -> torch.Tensor:
        """Inicializa con He Normal"""
        fan_in, fan_out = self.calculate_fan_in_fan_out(tensor)

        # Calcular fan según el modo
        if self.config.fan_mode == "fan_in":
            fan = fan_in
        elif self.config.fan_mode == "fan_out":
            fan = fan_out
        else:  # fan_avg
            fan = (fan_in + fan_out) / 2

        # Calcular desviación estándar
        std = math.sqrt(2.0 / fan)

        # Inicializar
        with torch.no_grad():
            tensor.normal_(0, std)

        return tensor


class OrthogonalInitializer(BaseWeightInitializer):
    """
    Inicializador Orthogonal mejorado
    """

    def initialize(self, tensor: torch.Tensor, layer_type: str = "linear") -> torch.Tensor:
        """Inicializa con distribución ortogonal"""
        if tensor.dim() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")

        # Reshape para operación ortogonal
        rows = tensor.size(0)
        cols = tensor.numel() // rows

        # Generar matriz ortogonal
        with torch.no_grad():
            flattened = tensor.view(rows, cols)
            if rows < cols:
                flattened.t_()

            # Generar matriz aleatoria
            q, r = torch.qr(torch.randn(rows, rows))
            d = torch.diag(r, 0)
            q *= d.sign()

            if rows < cols:
                q.t_()

            # Aplicar gain
            q *= self.config.orthogonal_gain

            # Reshape de vuelta
            tensor.copy_(q.view_as(tensor))

        return tensor


class SparseInitializer(BaseWeightInitializer):
    """
    Inicializador Sparse inteligente
    """

    def initialize(self, tensor: torch.Tensor, layer_type: str = "linear") -> torch.Tensor:
        """Inicializa con distribución sparse"""
        # Calcular número de elementos a mantener
        total_elements = tensor.numel()
        num_nonzero = int(total_elements * self.config.sparse_ratio)

        # Crear máscara sparse
        mask = torch.zeros_like(tensor)
        flat_mask = mask.view(-1)
        indices = torch.randperm(total_elements)[:num_nonzero]
        flat_mask[indices] = 1

        # Inicializar pesos
        fan_in, fan_out = self.calculate_fan_in_fan_out(tensor)
        fan = fan_in if self.config.fan_mode == "fan_in" else fan_out

        # Usar He initialization para los pesos no cero
        std = math.sqrt(2.0 / fan)

        with torch.no_grad():
            tensor.normal_(0, std)
            tensor *= mask

        return tensor


class AdaptiveInitializer(BaseWeightInitializer):
    """
    Inicializador adaptativo que selecciona el mejor método según el tipo de capa
    """

    def __init__(self, config: WeightInitializationConfig):
        super().__init__(config)
        self.initializers = {
            'xavier_uniform': XavierUniformInitializer(config),
            'xavier_normal': XavierNormalInitializer(config),
            'he_uniform': HeUniformInitializer(config),
            'he_normal': HeNormalInitializer(config),
            'orthogonal': OrthogonalInitializer(config),
            'sparse': SparseInitializer(config)
        }

    def initialize(self, tensor: torch.Tensor, layer_type: str = "linear") -> torch.Tensor:
        """Inicializa adaptativamente según el tipo de capa"""

        # Seleccionar método según el tipo de capa
        if layer_type in ['conv2d', 'conv3d']:
            method = 'he_uniform' if 'relu' in self.config.nonlinearity else 'xavier_uniform'
        elif layer_type in ['linear', 'fc']:
            method = 'xavier_uniform'
        elif layer_type in ['lstm', 'gru', 'rnn']:
            method = 'orthogonal'
        elif layer_type in ['embedding']:
            method = 'xavier_normal'
        else:
            method = self.config.method

        # Aplicar inicialización
        initializer = self.initializers[method]
        return initializer.initialize(tensor, layer_type)


class LayerSpecificInitializer:
    """
    Inicializador específico por capas con análisis de arquitectura
    """

    def __init__(self, config: WeightInitializationConfig):
        self.config = config
        self.adaptive_init = AdaptiveInitializer(config)
        self.layer_configs = self._create_layer_configs()

    def _create_layer_configs(self) -> Dict[str, WeightInitializationConfig]:
        """Crea configuraciones específicas para cada tipo de capa"""
        configs = {}

        # Configuración para capas convolucionales
        configs['conv'] = WeightInitializationConfig(
            method='he_uniform',
            fan_mode='fan_in',
            nonlinearity='relu',
            gain=math.sqrt(2.0)
        )

        # Configuración para capas lineales
        configs['linear'] = WeightInitializationConfig(
            method='xavier_uniform',
            fan_mode='fan_avg',
            nonlinearity='linear',
            gain=1.0
        )

        # Configuración para capas recurrentes
        configs['rnn'] = WeightInitializationConfig(
            method='orthogonal',
            orthogonal_gain=1.0
        )

        # Configuración para embeddings
        configs['embedding'] = WeightInitializationConfig(
            method='xavier_normal',
            fan_mode='fan_in',
            nonlinearity='linear',
            gain=1.0
        )

        # Configuración para capas de normalización
        configs['norm'] = WeightInitializationConfig(
            method='xavier_uniform',
            gain=0.1
        )

        return configs

    def initialize_layer(self, layer: nn.Module, layer_name: str = "") -> None:
        """Inicializa una capa específica"""

        # Determinar tipo de capa
        layer_type = self._get_layer_type(layer)

        # Obtener configuración específica
        if layer_type in self.layer_configs:
            layer_config = self.layer_configs[layer_type]
        else:
            layer_config = self.config

        # Crear inicializador para esta capa
        initializer = AdaptiveInitializer(layer_config)

        # Inicializar parámetros
        for name, param in layer.named_parameters():
            if 'weight' in name:
                initializer.initialize(param.data, layer_type)
            elif 'bias' in name:
                self._initialize_bias(param.data, layer_type)

    def _get_layer_type(self, layer: nn.Module) -> str:
        """Determina el tipo de capa"""
        layer_name = layer.__class__.__name__.lower()

        if 'conv' in layer_name:
            return 'conv'
        elif 'linear' in layer_name or 'fc' in layer_name:
            return 'linear'
        elif any(rnn_type in layer_name for rnn_type in ['lstm', 'gru', 'rnn']):
            return 'rnn'
        elif 'embedding' in layer_name:
            return 'embedding'
        elif any(norm_type in layer_name for norm_type in ['norm', 'bn', 'ln']):
            return 'norm'
        else:
            return 'linear'  # Default

    def _initialize_bias(self, bias: torch.Tensor, layer_type: str) -> None:
        """Inicializa el bias según el tipo de capa"""
        with torch.no_grad():
            if layer_type == 'conv':
                # Para convoluciones, inicializar con 0
                bias.zero_()
            elif layer_type == 'linear':
                # Para capas lineales, inicializar con 0
                bias.zero_()
            elif layer_type == 'rnn':
                # Para RNNs, inicializar con valores pequeños
                bias.uniform_(-0.1, 0.1)
            else:
                # Default: inicializar con 0
                bias.zero_()


class WeightInitializationManager:
    """
    Gestor principal de inicialización de pesos
    Coordina todas las técnicas de inicialización
    """

    def __init__(self, config: WeightInitializationConfig):
        self.config = config
        self.layer_initializer = LayerSpecificInitializer(config)
        self.initialization_history = []

    def initialize_model(self, model: nn.Module) -> None:
        """Inicializa todos los pesos del modelo"""
        logger.info("Iniciando inicialización de pesos del modelo")

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Solo módulos hoja
                self.layer_initializer.initialize_layer(module, name)

        logger.info("Inicialización de pesos completada")

    def initialize_layer(self, layer: nn.Module, layer_name: str = "") -> None:
        """Inicializa una capa específica"""
        self.layer_initializer.initialize_layer(layer, layer_name)

    def analyze_initialization_quality(self, model: nn.Module) -> Dict[str, float]:
        """Analiza la calidad de la inicialización"""
        analysis = {}

        for name, param in model.named_parameters():
            if 'weight' in name:
                # Calcular estadísticas
                weight_norm = param.data.norm().item()
                weight_std = param.data.std().item()
                weight_mean = param.data.mean().item()

                analysis[f"{name}_norm"] = weight_norm
                analysis[f"{name}_std"] = weight_std
                analysis[f"{name}_mean"] = weight_mean

        return analysis

    def get_initialization_summary(self) -> Dict:
        """Obtiene un resumen del estado de inicialización"""
        return {
            'method': self.config.method,
            'adaptive_init': self.config.adaptive_init,
            'layer_specific_init': self.config.layer_specific_init,
            'total_layers_initialized': len(self.initialization_history),
            'config': {
                'gain': self.config.gain,
                'fan_mode': self.config.fan_mode,
                'nonlinearity': self.config.nonlinearity,
                'sparse_ratio': self.config.sparse_ratio
            }
        }

# Funciones de utilidad


def initialize_weights_xavier_uniform(model: nn.Module, gain: float = 1.0) -> None:
    """Inicializa pesos usando Xavier Uniform"""
    config = WeightInitializationConfig(method='xavier_uniform', gain=gain)
    manager = WeightInitializationManager(config)
    manager.initialize_model(model)


def initialize_weights_he_uniform(model: nn.Module) -> None:
    """Inicializa pesos usando He Uniform"""
    config = WeightInitializationConfig(method='he_uniform')
    manager = WeightInitializationManager(config)
    manager.initialize_model(model)


def initialize_weights_orthogonal(model: nn.Module, gain: float = 1.0) -> None:
    """Inicializa pesos usando distribución ortogonal"""
    config = WeightInitializationConfig(method='orthogonal', orthogonal_gain=gain)
    manager = WeightInitializationManager(config)
    manager.initialize_model(model)


def initialize_weights_sparse(model: nn.Module, sparse_ratio: float = 0.1) -> None:
    """Inicializa pesos usando distribución sparse"""
    config = WeightInitializationConfig(method='sparse', sparse_ratio=sparse_ratio)
    manager = WeightInitializationManager(config)
    manager.initialize_model(model)


def initialize_weights_adaptive(model: nn.Module,
                                adaptive: bool = True,
                                layer_specific: bool = True) -> None:
    """Inicializa pesos usando método adaptativo"""
    config = WeightInitializationConfig(
        adaptive_init=adaptive,
        layer_specific_init=layer_specific
    )
    manager = WeightInitializationManager(config)
    manager.initialize_model(model)


def compare_initialization_methods(model_class: type,
                                   input_shape: Tuple[int, ...],
                                   methods: List[str] = None) -> Dict[str, Dict]:
    """Compara diferentes métodos de inicialización"""

    if methods is None:
        methods = ['xavier_uniform', 'he_uniform', 'orthogonal', 'sparse']

    results = {}

    for method in methods:
        # Crear modelo
        model = model_class()

        # Inicializar pesos
        config = WeightInitializationConfig(method=method)
        manager = WeightInitializationManager(config)
        manager.initialize_model(model)

        # Analizar calidad
        analysis = manager.analyze_initialization_quality(model)
        results[method] = analysis

    return results


# Exportar clases y funciones principales
__all__ = [
    'WeightInitializationConfig',
    'BaseWeightInitializer',
    'XavierUniformInitializer',
    'XavierNormalInitializer',
    'HeUniformInitializer',
    'HeNormalInitializer',
    'OrthogonalInitializer',
    'SparseInitializer',
    'AdaptiveInitializer',
    'LayerSpecificInitializer',
    'WeightInitializationManager',
    'initialize_weights_xavier_uniform',
    'initialize_weights_he_uniform',
    'initialize_weights_orthogonal',
    'initialize_weights_sparse',
    'initialize_weights_adaptive',
    'compare_initialization_methods'
]

logger.info("RFEN3_RN_4 - Métodos de Inicialización de Pesos Inteligentes cargados correctamente")
