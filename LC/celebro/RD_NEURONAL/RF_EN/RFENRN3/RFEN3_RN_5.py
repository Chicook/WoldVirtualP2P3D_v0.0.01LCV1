"""
RFEN3_RN_5 - Técnicas de Optimización de Gradientes Avanzadas
Implementación de métodos modernos para optimización y manipulación de gradientes
Incluye: Gradient Accumulation, Gradient Checkpointing, Mixed Precision, Gradient Scaling
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.cuda.amp as amp
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
import math
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import contextlib

logger = logging.getLogger(__name__)


@dataclass
class GradientOptimizationConfig:
    """Configuración para optimización de gradientes"""
    accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    gradient_scaling: bool = True
    max_grad_norm: float = 1.0
    gradient_clipping: str = "norm"  # norm, value, adaptive
    loss_scaling: float = 1.0
    fp16_loss_scale: float = 1024.0
    gradient_accumulation_schedule: str = "linear"  # linear, cosine, exponential
    checkpoint_strategy: str = "selective"  # selective, uniform, adaptive


class GradientAccumulator:
    """
    Acumulador de gradientes inteligente con scheduling adaptativo
    """

    def __init__(self, config: GradientOptimizationConfig):
        self.config = config
        self.accumulation_steps = config.accumulation_steps
        self.current_step = 0
        self.gradient_history = []
        self.accumulation_schedule = self._create_schedule()

    def _create_schedule(self) -> List[int]:
        """Crea un schedule de acumulación de gradientes"""
        if self.config.gradient_accumulation_schedule == "linear":
            return [self.accumulation_steps] * 1000  # Schedule constante
        elif self.config.gradient_accumulation_schedule == "cosine":
            # Schedule coseno que aumenta gradualmente
            steps = []
            for i in range(1000):
                factor = 0.5 * (1 + math.cos(math.pi * i / 1000))
                steps.append(int(self.accumulation_steps * (1 + factor)))
            return steps
        elif self.config.gradient_accumulation_schedule == "exponential":
            # Schedule exponencial
            steps = []
            for i in range(1000):
                factor = math.exp(-i / 200)
                steps.append(int(self.accumulation_steps * (1 + factor)))
            return steps
        else:
            return [self.accumulation_steps] * 1000

    def should_accumulate(self) -> bool:
        """Determina si se debe acumular gradientes en este paso"""
        if self.current_step < len(self.accumulation_schedule):
            required_steps = self.accumulation_schedule[self.current_step]
        else:
            required_steps = self.accumulation_steps

        return (self.current_step % required_steps) != (required_steps - 1)

    def step(self) -> None:
        """Incrementa el contador de pasos"""
        self.current_step += 1

    def get_current_accumulation_steps(self) -> int:
        """Obtiene el número de pasos de acumulación actual"""
        if self.current_step < len(self.accumulation_schedule):
            return self.accumulation_schedule[self.current_step]
        return self.accumulation_steps


class GradientCheckpointer:
    """
    Sistema de Gradient Checkpointing inteligente
    """

    def __init__(self, config: GradientOptimizationConfig):
        self.config = config
        self.checkpoint_strategy = config.checkpoint_strategy
        self.memory_savings = 0.0
        self.checkpoint_history = []

    def should_checkpoint(self, module: nn.Module, layer_index: int, total_layers: int) -> bool:
        """Determina si una capa debe ser checkpointed"""

        if self.checkpoint_strategy == "uniform":
            # Checkpoint uniforme cada N capas
            checkpoint_interval = max(1, total_layers // 4)
            return layer_index % checkpoint_interval == 0

        elif self.checkpoint_strategy == "selective":
            # Checkpoint selectivo basado en el tipo de capa
            layer_name = module.__class__.__name__.lower()
            if any(layer_type in layer_name for layer_type in ['conv', 'linear', 'attention']):
                return True
            return False

        elif self.checkpoint_strategy == "adaptive":
            # Checkpoint adaptativo basado en el uso de memoria
            if hasattr(module, 'weight'):
                param_size = module.weight.numel() * module.weight.element_size()
                # Checkpoint si la capa es grande
                return param_size > 1024 * 1024  # 1MB threshold

        return False

    def checkpoint_module(self, module: nn.Module, *args, **kwargs):
        """Aplica checkpointing a un módulo"""
        return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)

    def estimate_memory_savings(self, model: nn.Module) -> float:
        """Estima el ahorro de memoria con checkpointing"""
        total_params = sum(p.numel() for p in model.parameters())
        checkpointed_params = 0

        for i, module in enumerate(model.modules()):
            if self.should_checkpoint(module, i, len(list(model.modules()))):
                checkpointed_params += sum(p.numel() for p in module.parameters())

        savings_ratio = checkpointed_params / total_params
        self.memory_savings = savings_ratio
        return savings_ratio


class MixedPrecisionManager:
    """
    Gestor de Mixed Precision Training con escalado automático de pérdida
    """

    def __init__(self, config: GradientOptimizationConfig):
        self.config = config
        # Usar la nueva API de GradScaler y sólo activarlo cuando CUDA esté disponible.
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        self.loss_scale_history = []
        self.scale_update_frequency = 100
        self.scale_update_counter = 0

    def autocast(self):
        """Context manager para autocast"""
        if self.scaler:
            return amp.autocast()
        else:
            return contextlib.nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Escala la pérdida para mixed precision"""
        if self.scaler:
            return self.scaler.scale(loss)
        return loss

    def unscale_gradients(self, optimizer: optim.Optimizer) -> None:
        """Desescala los gradientes"""
        if self.scaler:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: optim.Optimizer) -> None:
        """Realiza un paso del optimizador"""
        if self.scaler:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Actualiza el scaler"""
        if self.scaler:
            self.scaler.update()
            self.scale_update_counter += 1

            # Registrar escala actual
            current_scale = self.scaler.get_scale()
            self.loss_scale_history.append(current_scale)

            if len(self.loss_scale_history) > 1000:
                self.loss_scale_history.pop(0)

    def get_current_scale(self) -> float:
        """Obtiene la escala actual"""
        if self.scaler:
            return self.scaler.get_scale()
        return 1.0


class AdaptiveGradientClipper:
    """
    Clipper de gradientes adaptativo que ajusta automáticamente la norma máxima
    """

    def __init__(self, config: GradientOptimizationConfig):
        self.config = config
        self.max_norm = config.max_grad_norm
        self.gradient_norm_history = []
        self.adaptation_rate = 0.1
        self.min_norm = 0.1
        self.max_norm_limit = 10.0

    def clip_gradients(self, model: nn.Module) -> float:
        """Aplica clipping de gradientes"""

        if self.config.gradient_clipping == "norm":
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.max_norm
            )
        elif self.config.gradient_clipping == "value":
            torch.nn.utils.clip_grad_value_(
                model.parameters(), self.max_norm
            )
            total_norm = self._calculate_gradient_norm(model)
        elif self.config.gradient_clipping == "adaptive":
            total_norm = self._adaptive_clip(model)
        else:
            total_norm = self._calculate_gradient_norm(model)

        # Actualizar historial
        self.gradient_norm_history.append(total_norm.item())
        if len(self.gradient_norm_history) > 100:
            self.gradient_norm_history.pop(0)

        # Adaptar norma máxima
        self._adapt_max_norm()

        return total_norm

    def _calculate_gradient_norm(self, model: nn.Module) -> torch.Tensor:
        """Calcula la norma de los gradientes"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return torch.tensor(total_norm ** (1. / 2))

    def _adaptive_clip(self, model: nn.Module) -> torch.Tensor:
        """Aplica clipping adaptativo"""
        if len(self.gradient_norm_history) < 10:
            return torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

        # Calcular percentil 95 de la historia
        recent_norms = self.gradient_norm_history[-10:]
        adaptive_norm = np.percentile(recent_norms, 95)

        # Usar la norma adaptativa
        return torch.nn.utils.clip_grad_norm_(model.parameters(), adaptive_norm)

    def _adapt_max_norm(self) -> None:
        """Adapta la norma máxima basado en el historial"""
        if len(self.gradient_norm_history) < 20:
            return

        recent_norms = self.gradient_norm_history[-20:]
        avg_norm = np.mean(recent_norms)

        # Ajustar norma máxima
        if avg_norm > self.max_norm * 1.5:
            self.max_norm = min(self.max_norm * (1 + self.adaptation_rate), self.max_norm_limit)
        elif avg_norm < self.max_norm * 0.5:
            self.max_norm = max(self.max_norm * (1 - self.adaptation_rate), self.min_norm)


class GradientOptimizationManager:
    """
    Gestor principal de optimización de gradientes
    Coordina todas las técnicas de optimización de gradientes
    """

    def __init__(self, config: GradientOptimizationConfig):
        self.config = config
        self.accumulator = GradientAccumulator(config)
        self.checkpointer = GradientCheckpointer(config)
        self.mixed_precision = MixedPrecisionManager(config)
        self.gradient_clipper = AdaptiveGradientClipper(config)

        self.optimization_stats = {
            'total_steps': 0,
            'accumulation_steps': 0,
            'checkpoint_savings': 0.0,
            'gradient_norms': [],
            'loss_scales': []
        }

    def initialize_model(self, model: nn.Module) -> None:
        """Inicializa el modelo con técnicas de optimización"""

        # Aplicar gradient checkpointing
        if self.config.gradient_checkpointing:
            self._apply_checkpointing(model)

        # Estimar ahorro de memoria
        if self.config.gradient_checkpointing:
            savings = self.checkpointer.estimate_memory_savings(model)
            self.optimization_stats['checkpoint_savings'] = savings
            logger.info(f"Gradient checkpointing estimado: {savings:.2%} ahorro de memoria")

    def _apply_checkpointing(self, model: nn.Module) -> None:
        """Aplica gradient checkpointing al modelo"""
        modules = list(model.modules())

        for i, module in enumerate(modules):
            if self.checkpointer.should_checkpoint(module, i, len(modules)):
                # Reemplazar forward con checkpointed version
                original_forward = module.forward

                def checkpointed_forward(*args, **kwargs):
                    return self.checkpointer.checkpoint_module(module, *args, **kwargs)

                module.forward = checkpointed_forward

    def training_step(self, model: nn.Module, optimizer: optim.Optimizer,
                      loss_fn: Callable, inputs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Realiza un paso de entrenamiento optimizado"""

        stats = {}

        # Determinar si acumular gradientes
        should_accumulate = self.accumulator.should_accumulate()

        with self.mixed_precision.autocast():
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Escalar pérdida para mixed precision
            scaled_loss = self.mixed_precision.scale_loss(loss)

            # Escalar por pasos de acumulación
            if should_accumulate:
                scaled_loss = scaled_loss / self.accumulator.get_current_accumulation_steps()

        # Backward pass
        scaled_loss.backward()

        if not should_accumulate:
            # Aplicar clipping de gradientes
            grad_norm = self.gradient_clipper.clip_gradients(model)

            # Desescalar gradientes para mixed precision
            self.mixed_precision.unscale_gradients(optimizer)

            # Paso del optimizador
            self.mixed_precision.step(optimizer)
            optimizer.zero_grad()

            # Actualizar scaler
            self.mixed_precision.update()

            # Actualizar estadísticas
            self.optimization_stats['total_steps'] += 1
            self.optimization_stats['gradient_norms'].append(grad_norm.item())
            self.optimization_stats['loss_scales'].append(self.mixed_precision.get_current_scale())

        # Actualizar contador de acumulación
        self.accumulator.step()

        # Preparar estadísticas
        stats.update({
            'loss': loss.item(),
            'scaled_loss': scaled_loss.item(),
            'gradient_norm': grad_norm.item() if not should_accumulate else 0.0,
            'loss_scale': self.mixed_precision.get_current_scale(),
            'accumulation_steps': self.accumulator.get_current_accumulation_steps(),
            'should_accumulate': should_accumulate
        })

        return stats

    def get_optimization_summary(self) -> Dict:
        """Obtiene un resumen del estado de optimización"""
        return {
            'config': {
                'accumulation_steps': self.config.accumulation_steps,
                'gradient_checkpointing': self.config.gradient_checkpointing,
                'mixed_precision': self.config.mixed_precision,
                'gradient_scaling': self.config.gradient_scaling,
                'max_grad_norm': self.config.max_grad_norm
            },
            'stats': self.optimization_stats,
            'current_settings': {
                'accumulation_steps': self.accumulator.get_current_accumulation_steps(),
                'max_grad_norm': self.gradient_clipper.max_norm,
                'loss_scale': self.mixed_precision.get_current_scale(),
                'memory_savings': self.optimization_stats['checkpoint_savings']
            }
        }

# Funciones de utilidad


def enable_gradient_checkpointing(model: nn.Module, strategy: str = "selective") -> None:
    """Habilita gradient checkpointing en un modelo"""
    config = GradientOptimizationConfig(gradient_checkpointing=True, checkpoint_strategy=strategy)
    manager = GradientOptimizationManager(config)
    manager.initialize_model(model)


def setup_mixed_precision_training(model: nn.Module, optimizer: optim.Optimizer) -> MixedPrecisionManager:
    """Configura mixed precision training"""
    config = GradientOptimizationConfig(mixed_precision=True)
    return MixedPrecisionManager(config)


def analyze_gradient_flow(model: nn.Module) -> Dict[str, float]:
    """Analiza el flujo de gradientes en el modelo"""
    gradient_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm().item()
            gradient_norms[name] = grad_norm

    return gradient_norms


def optimize_gradient_memory(model: nn.Module,
                             accumulation_steps: int = 4,
                             checkpointing: bool = True) -> GradientOptimizationManager:
    """Configura optimización de memoria para gradientes"""
    config = GradientOptimizationConfig(
        accumulation_steps=accumulation_steps,
        gradient_checkpointing=checkpointing,
        mixed_precision=True
    )

    manager = GradientOptimizationManager(config)
    manager.initialize_model(model)

    return manager


def benchmark_gradient_optimization(model: nn.Module,
                                    optimizer: optim.Optimizer,
                                    loss_fn: Callable,
                                    data_loader: torch.utils.data.DataLoader,
                                    config: GradientOptimizationConfig) -> Dict[str, float]:
    """Realiza benchmark de técnicas de optimización de gradientes"""

    manager = GradientOptimizationManager(config)
    manager.initialize_model(model)

    # Medir tiempo y memoria
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    total_loss = 0.0
    steps = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        if batch_idx >= 10:  # Limitar a 10 batches para benchmark
            break

        stats = manager.training_step(model, optimizer, loss_fn, inputs, targets)
        total_loss += stats['loss']
        steps += 1

    end_time = time.time()
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    return {
        'total_time': end_time - start_time,
        'avg_time_per_step': (end_time - start_time) / steps,
        'memory_used': end_memory - start_memory,
        'avg_loss': total_loss / steps,
        'optimization_summary': manager.get_optimization_summary()
    }


# Exportar clases y funciones principales
__all__ = [
    'GradientOptimizationConfig',
    'GradientAccumulator',
    'GradientCheckpointer',
    'MixedPrecisionManager',
    'AdaptiveGradientClipper',
    'GradientOptimizationManager',
    'enable_gradient_checkpointing',
    'setup_mixed_precision_training',
    'analyze_gradient_flow',
    'optimize_gradient_memory',
    'benchmark_gradient_optimization'
]

logger.info("RFEN3_RN_5 - Técnicas de Optimización de Gradientes Avanzadas cargadas correctamente")
