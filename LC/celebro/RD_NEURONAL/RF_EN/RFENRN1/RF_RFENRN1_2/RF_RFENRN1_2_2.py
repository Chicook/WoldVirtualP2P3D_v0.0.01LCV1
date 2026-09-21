"""
RF_RFENRN1_2_2.py - Gestor de Regularización Avanzada
=====================================================

Implementa técnicas avanzadas de regularización para mejorar la generalización
y obtener pesos de mayor calidad. Incluye SAM, SWA, Label Smoothing, Dropout
estructurado y técnicas de peso decay adaptativo.

Características:
- SAM (Sharpness-Aware Minimization) para pesos planos
- SWA (Stochastic Weight Averaging) para soluciones estables
- Label Smoothing para mejorar generalización
- Dropout estructurado y adaptativo
- Weight Decay adaptativo por capas

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
import copy
from dataclasses import dataclass

logger = logging.getLogger('RFENRN1.RF_RFENRN1_2_2')


@dataclass
class RegularizationConfig:
    """Configuración para técnicas de regularización"""
    use_sam: bool = True
    sam_rho: float = 0.05
    use_swa: bool = True
    swa_start: int = 75
    swa_freq: int = 5
    swa_lr: float = 0.05
    use_label_smoothing: bool = True
    label_smoothing_alpha: float = 0.1
    use_structured_dropout: bool = True
    dropout_rate: float = 0.1
    adaptive_dropout: bool = True
    use_weight_decay: bool = True
    weight_decay: float = 1e-4
    adaptive_weight_decay: bool = True
    layer_specific_decay: bool = True


class RegularizationManager:
    """
    Gestor de técnicas de regularización avanzada.

    Implementa SAM, SWA, Label Smoothing y otras técnicas para
    mejorar la calidad de los pesos neuronales.
    """

    def __init__(self, config: Optional[RegularizationConfig] = None):
        """
        Inicializa el gestor de regularización.

        Args:
            config: Configuración de regularización (opcional)
        """
        self.config = config or RegularizationConfig()
        self.swa_model = None
        self.swa_n = 0
        self.swa_enabled = False
        self.sam_weights = None
        self.regularization_history = []
        self.metrics = {
            'sharpness': 0.0,
            'weight_stability': 0.0,
            'generalization_gap': 0.0,
            'model_complexity': 0.0
        }

        logger.info("RegularizationManager inicializado")

    def apply_sam_regularization(self, model: nn.Module, loss_fn, closure) -> torch.Tensor:
        """
        Aplica SAM (Sharpness-Aware Minimization) para encontrar pesos planos.

        Args:
            model: Modelo PyTorch
            loss_fn: Función de pérdida
            closure: Cierre que calcula la pérdida

        Returns:
            Pérdida SAM
        """
        if not self.config.use_sam:
            return loss_fn()

        # Guardar pesos originales
        e_w = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                e_w.append(param.data.clone())

        # Calcular gradiente
        first_loss = loss_fn()
        first_loss.backward()

        # Calcular perturbación epsilon
        grads = [param.grad.data for param in model.parameters() if param.requires_grad]
        grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))

        eps_sam = [self.config.sam_rho * g / (grad_norm + 1e-12) for g in grads]

        # Aplicar perturbación
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.add_(eps_sam[idx])
                idx += 1

        # Calcular pérdida perturbada
        second_loss = loss_fn()

        # Restaurar pesos
        idx = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(e_w[idx])
                idx += 1

        # Invertir gradiente para paso ascendente
        second_loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                param.grad = -param.grad

        return second_loss

    def enable_swa(self, model: nn.Module) -> None:
        """
        Habilita Stochastic Weight Averaging.

        Args:
            model: Modelo PyTorch
        """
        self.swa_model = copy.deepcopy(model)
        self.swa_n = 0
        self.swa_enabled = True

        for swa_param, model_param in zip(self.swa_model.parameters(), model.parameters()):
            swa_param.data = model_param.data.clone()

        logger.info("SWA habilitado")

    def update_swa(self, model: nn.Module, step: int) -> None:
        """
        Actualiza el modelo SWA.

        Args:
            model: Modelo PyTorch
            step: Paso actual de entrenamiento
        """
        if not self.config.use_swa or not self.swa_enabled:
            return

        if step >= self.config.swa_start and step % self.config.swa_freq == 0:
            for swa_param, model_param in zip(self.swa_model.parameters(), model.parameters()):
                swa_param.data = (
                    self.swa_n * swa_param.data + model_param.data
                ) / (self.swa_n + 1)

            self.swa_n += 1

    def apply_swa_weights(self, model: nn.Module) -> None:
        """
        Aplica pesos SWA al modelo.

        Args:
            model: Modelo PyTorch
        """
        if self.swa_model is not None and self.swa_n > 0:
            for param, swa_param in zip(model.parameters(), self.swa_model.parameters()):
                param.data.copy_(swa_param.data)

            logger.info(f"Pesos SWA aplicados (n={self.swa_n})")

    def apply_label_smoothing(self, target: torch.Tensor, n_classes: int) -> torch.Tensor:
        """
        Aplica Label Smoothing.

        Args:
            target: Targets originales
            n_classes: Número de clases

        Returns:
            Targets suavizados
        """
        if not self.config.use_label_smoothing:
            return target

        smooth_target = target.clone().float()
        smooth_target *= (1 - self.config.label_smoothing_alpha)
        smooth_target += self.config.label_smoothing_alpha / n_classes

        return smooth_target

    def apply_structured_dropout(self, x: torch.Tensor, layer_type: str = 'linear') -> torch.Tensor:
        """
        Aplica Dropout Estructurado.

        Args:
            x: Tensor de entrada
            layer_type: Tipo de capa

        Returns:
            Tensor con dropout aplicado
        """
        if not self.config.use_structured_dropout:
            return x

        if self.config.adaptive_dropout:
            # Dropout adaptativo basado en varianza
            var = torch.var(x, dim=-1, keepdim=True)
            dropout_rate = self.config.dropout_rate * (1 - torch.sigmoid(var))
            return F.dropout(x, p=dropout_rate.mean().item(), training=self.training)
        else:
            return F.dropout(x, p=self.config.dropout_rate, training=self.training)

    def calculate_sharpness(self, model: nn.Module, loss_fn, data_loader) -> float:
        """
        Calcula la sharpness del mínimo (para SAM).

        Args:
            model: Modelo PyTorch
            loss_fn: Función de pérdida
            data_loader: DataLoader

        Returns:
            Valor de sharpness
        """
        if not self.config.use_sam:
            return 0.0

        # Esto sería calculado durante el entrenamiento SAM
        # Aquí devolvemos un placeholder
        return self.metrics.get('sharpness', 0.0)

    def calculate_weight_stability(self, model: nn.Module, previous_weights: Dict) -> float:
        """
        Calcula la estabilidad de los pesos.

        Args:
            model: Modelo PyTorch
            previous_weights: Pesos anteriores

        Returns:
            Score de estabilidad
        """
        total_change = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if name in previous_weights:
                change = torch.norm(param.data - previous_weights[name]).item()
                total_change += change
                param_count += 1

        if param_count > 0:
            avg_change = total_change / param_count
            stability = 1.0 / (1.0 + avg_change)
            self.metrics['weight_stability'] = stability
            return stability

        return 0.0

    def apply_adaptive_weight_decay(self, optimizer, step: int, total_steps: int) -> None:
        """
        Aplica Weight Decay adaptativo basado en el progreso del entrenamiento.

        Args:
            optimizer: Optimizador
            step: Paso actual
            total_steps: Total de pasos
        """
        if not self.config.adaptive_weight_decay:
            return

        # Aumentar weight decay gradualmente
        progress = step / total_steps
        adaptive_decay = self.config.weight_decay * (1.0 + progress)

        for group in optimizer.param_groups:
            if 'weight_decay' in group:
                group['weight_decay'] = adaptive_decay

    def get_metrics(self) -> Dict[str, float]:
        """
        Obtiene las métricas de regularización.

        Returns:
            Diccionario con métricas
        """
        return self.metrics.copy()

    def reset_swa(self) -> None:
        """Reinicia el modelo SWA."""
        self.swa_model = None
        self.swa_n = 0
        self.swa_enabled = False
        logger.info("SWA reiniciado")
