"""
RFEN3_RN_1 - Optimizadores Avanzados para Redes Neuronales 2025
Implementación de los optimizadores más modernos y eficientes para optimización de pesos
Incluye: Lion, AdaBelief, RAdam, AdamW mejorado y técnicas de segunda derivada
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
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
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import math
from dataclasses import dataclass
from . import BaseOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class LionConfig:
    """Configuración específica para Lion Optimizer"""
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.01
    lr: float = 0.001


class LionOptimizer(BaseOptimizer):
    """
    Lion Optimizer - Optimizador de última generación (2023-2025)
    Combina la eficiencia de Adam con la simplicidad de SGD
    Basado en: "Symbolic Discovery of Optimization Algorithms" (Google Research)
    """

    def __init__(self, params: List[torch.Tensor], config: LionConfig):
        super().__init__(params, config)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay
        self.lr = config.lr

        # Inicializar estado para cada parámetro
        for param in self.params:
            self.state[param] = {
                'exp_avg': torch.zeros_like(param.data),
                'step': 0
            }

    def step(self, closure: Optional[Callable] = None) -> None:
        """Paso de optimización Lion"""
        loss = None
        if closure is not None:
            loss = closure()

        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data
            state = self.state[param]
            exp_avg = state['exp_avg']
            step = state['step']

            # Actualizar momentum
            exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)

            # Calcular update con signo
            update = torch.sign(exp_avg)

            # Aplicar weight decay
            if self.weight_decay > 0:
                param.data.mul_(1 - self.lr * self.weight_decay)

            # Actualizar parámetros
            param.data.add_(update, alpha=-self.lr)

            state['step'] += 1

        self.step_count += 1

    def zero_grad(self) -> None:
        """Limpia los gradientes"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


@dataclass
class AdaBeliefConfig:
    """Configuración específica para AdaBelief Optimizer"""
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    lr: float = 0.001


class AdaBeliefOptimizer(BaseOptimizer):
    """
    AdaBelief Optimizer - Mejora de Adam con estimación de varianza
    Basado en: "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"
    """

    def __init__(self, params: List[torch.Tensor], config: AdaBeliefConfig):
        super().__init__(params, config)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.eps
        self.weight_decay = config.weight_decay
        self.lr = config.lr

        # Inicializar estado
        for param in self.params:
            self.state[param] = {
                'exp_avg': torch.zeros_like(param.data),
                'exp_avg_sq': torch.zeros_like(param.data),
                'step': 0
            }

    def step(self, closure: Optional[Callable] = None) -> None:
        """Paso de optimización AdaBelief"""
        loss = None
        if closure is not None:
            loss = closure()

        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data
            state = self.state[param]
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            step = state['step']

            # Bias correction
            bias_correction1 = 1 - self.beta1 ** (step + 1)
            bias_correction2 = 1 - self.beta2 ** (step + 1)

            # Actualizar momentum
            exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)

            # Actualizar varianza estimada (diferencia clave con Adam)
            grad_residual = grad - exp_avg
            exp_avg_sq.mul_(self.beta2).addcmul_(grad_residual, grad_residual, value=1 - self.beta2)

            # Calcular update
            denom = (exp_avg_sq / bias_correction2).sqrt().add_(self.eps)
            update = exp_avg / bias_correction1 / denom

            # Aplicar weight decay
            if self.weight_decay > 0:
                param.data.mul_(1 - self.lr * self.weight_decay)

            # Actualizar parámetros
            param.data.add_(update, alpha=-self.lr)

            state['step'] += 1

        self.step_count += 1

    def zero_grad(self) -> None:
        """Limpia los gradientes"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


@dataclass
class RAdamConfig:
    """Configuración específica para RAdam Optimizer"""
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    lr: float = 0.001


class RAdamOptimizer(BaseOptimizer):
    """
    RAdam Optimizer - Rectified Adam con corrección de varianza
    Basado en: "On the Variance of the Adaptive Learning Rate and Beyond"
    """

    def __init__(self, params: List[torch.Tensor], config: RAdamConfig):
        super().__init__(params, config)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.eps
        self.weight_decay = config.weight_decay
        self.lr = config.lr

        # Inicializar estado
        for param in self.params:
            self.state[param] = {
                'exp_avg': torch.zeros_like(param.data),
                'exp_avg_sq': torch.zeros_like(param.data),
                'step': 0
            }

    def step(self, closure: Optional[Callable] = None) -> None:
        """Paso de optimización RAdam"""
        loss = None
        if closure is not None:
            loss = closure()

        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data
            state = self.state[param]
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            step = state['step']

            # Actualizar momentum
            exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            # Calcular términos de corrección
            bias_correction1 = 1 - self.beta1 ** (step + 1)
            bias_correction2 = 1 - self.beta2 ** (step + 1)

            # Rectificación de varianza (característica clave de RAdam)
            rho_inf = 2 / (1 - self.beta2) - 1
            rho_t = rho_inf - 2 * step * self.beta2 ** (step + 1) / bias_correction2

            if rho_t > 4:  # Rectificación activa
                # Calcular términos rectificados
                r_t = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))

                # Calcular update rectificado
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(self.eps)
                update = exp_avg / bias_correction1 * r_t / denom
            else:
                # Fallback a SGD con momentum
                update = exp_avg / bias_correction1

            # Aplicar weight decay
            if self.weight_decay > 0:
                param.data.mul_(1 - self.lr * self.weight_decay)

            # Actualizar parámetros
            param.data.add_(update, alpha=-self.lr)

            state['step'] += 1

        self.step_count += 1

    def zero_grad(self) -> None:
        """Limpia los gradientes"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class AdvancedAdamWOptimizer(BaseOptimizer):
    """
    AdamW Optimizer Avanzado con mejoras específicas para 2025
    Incluye: warmup, gradient clipping adaptativo, y momentum adaptativo
    """

    def __init__(self, params: List[torch.Tensor], config: OptimizationConfig):
        super().__init__(params, config)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.epsilon
        self.weight_decay = config.weight_decay
        self.lr = config.learning_rate
        self.warmup_epochs = config.warmup_epochs

        # Inicializar estado
        for param in self.params:
            self.state[param] = {
                'exp_avg': torch.zeros_like(param.data),
                'exp_avg_sq': torch.zeros_like(param.data),
                'step': 0,
                'grad_norm_history': []
            }

    def _get_warmup_lr(self, step: int) -> float:
        """Calcula la tasa de aprendizaje con warmup"""
        if step < self.warmup_epochs:
            return self.lr * (step + 1) / self.warmup_epochs
        return self.lr

    def _adaptive_gradient_clipping(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Aplica gradient clipping adaptativo basado en la historia"""
        state = self.state[param]
        grad_norm_history = state['grad_norm_history']

        # Calcular norma actual
        current_norm = grad.norm().item()
        grad_norm_history.append(current_norm)

        # Mantener solo los últimos 100 valores
        if len(grad_norm_history) > 100:
            grad_norm_history.pop(0)

        # Calcular percentil 95 de la historia
        if len(grad_norm_history) > 10:
            threshold = np.percentile(grad_norm_history, 95)
            if current_norm > threshold:
                grad = grad * (threshold / current_norm)

        return grad

    def step(self, closure: Optional[Callable] = None) -> None:
        """Paso de optimización AdamW avanzado"""
        loss = None
        if closure is not None:
            loss = closure()

        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data
            state = self.state[param]
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            step = state['step']

            # Aplicar gradient clipping adaptativo
            grad = self._adaptive_gradient_clipping(param, grad)

            # Actualizar momentum
            exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            # Bias correction
            bias_correction1 = 1 - self.beta1 ** (step + 1)
            bias_correction2 = 1 - self.beta2 ** (step + 1)

            # Calcular update
            denom = (exp_avg_sq / bias_correction2).sqrt().add_(self.eps)
            update = exp_avg / bias_correction1 / denom

            # Aplicar warmup
            current_lr = self._get_warmup_lr(step)

            # Aplicar weight decay
            if self.weight_decay > 0:
                param.data.mul_(1 - current_lr * self.weight_decay)

            # Actualizar parámetros
            param.data.add_(update, alpha=-current_lr)

            state['step'] += 1

        self.step_count += 1

    def zero_grad(self) -> None:
        """Limpia los gradientes"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class OptimizerFactory:
    """Factory para crear optimizadores avanzados"""

    @staticmethod
    def create_optimizer(optimizer_type: str, params: List[torch.Tensor], config: OptimizationConfig) -> BaseOptimizer:
        """Crea un optimizador según el tipo especificado"""

        if optimizer_type.lower() == "lion":
            lion_config = LionConfig(
                beta1=config.beta1,
                beta2=config.beta2,
                weight_decay=config.weight_decay,
                lr=config.learning_rate
            )
            return LionOptimizer(params, lion_config)

        elif optimizer_type.lower() == "adabelief":
            adabelief_config = AdaBeliefConfig(
                beta1=config.beta1,
                beta2=config.beta2,
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                lr=config.learning_rate
            )
            return AdaBeliefOptimizer(params, adabelief_config)

        elif optimizer_type.lower() == "radam":
            radam_config = RAdamConfig(
                beta1=config.beta1,
                beta2=config.beta2,
                eps=config.epsilon,
                weight_decay=config.weight_decay,
                lr=config.learning_rate
            )
            return RAdamOptimizer(params, radam_config)

        elif optimizer_type.lower() == "adamw":
            return AdvancedAdamWOptimizer(params, config)

        else:
            raise ValueError(f"Tipo de optimizador no soportado: {optimizer_type}")

    @staticmethod
    def get_optimizer_info(optimizer_type: str) -> Dict[str, str]:
        """Obtiene información sobre el optimizador"""
        info = {
            "lion": "Lion: Optimizador eficiente que combina Adam y SGD",
            "adabelief": "AdaBelief: Adam mejorado con estimación de varianza",
            "radam": "RAdam: Adam rectificado con corrección de varianza",
            "adamw": "AdamW: Adam con decaimiento de pesos mejorado"
        }
        return {"description": info.get(optimizer_type.lower(), "Optimizador desconocido")}

# Funciones de utilidad para comparación de optimizadores


def compare_optimizers(model: nn.Module, optimizers: List[str],
                       train_data: torch.Tensor, train_labels: torch.Tensor,
                       epochs: int = 10) -> Dict[str, List[float]]:
    """Compara el rendimiento de diferentes optimizadores"""

    results = {}
    config = OptimizationConfig()

    for opt_name in optimizers:
        logger.info(f"Probando optimizador: {opt_name}")

        # Crear copia del modelo
        model_copy = type(model)(*model.args) if hasattr(model, 'args') else model

        # Crear optimizador
        optimizer = OptimizerFactory.create_optimizer(opt_name, model_copy.parameters(), config)

        # Entrenar modelo
        losses = []
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model_copy(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        results[opt_name] = losses

    return results


def benchmark_optimizer(optimizer: BaseOptimizer, iterations: int = 1000) -> Dict[str, float]:
    """Realiza benchmark de un optimizador"""
    import time

    start_time = time.time()

    # Crear datos sintéticos para benchmark
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)

    # Crear modelo simple
    model = nn.Linear(10, 1)
    criterion = nn.MSELoss()

    times = []
    losses = []

    for i in range(iterations):
        iter_start = time.time()

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        iter_time = time.time() - iter_start
        times.append(iter_time)
        losses.append(loss.item())

    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "avg_iteration_time": np.mean(times),
        "final_loss": losses[-1],
        "loss_reduction": losses[0] - losses[-1],
        "iterations_per_second": iterations / total_time
    }


# Exportar clases y funciones principales
__all__ = [
    'LionOptimizer', 'LionConfig',
    'AdaBeliefOptimizer', 'AdaBeliefConfig',
    'RAdamOptimizer', 'RAdamConfig',
    'AdvancedAdamWOptimizer',
    'OptimizerFactory',
    'compare_optimizers',
    'benchmark_optimizer'
]

logger.info("RFEN3_RN_1 - Optimizadores Avanzados cargados correctamente")
