"""
RF_RFENRN1_4_7.py - Gestor de Pruning y Sparse Optimization
===========================================================

Implementa técnicas avanzadas de pruning y optimización sparse para redes neuronales
de aprendizaje por refuerzo. Incluye pruning estructurado, no estructurado y técnicas
de compresión de modelos.

Características:
- Pruning estructurado y no estructurado
- Magnitude-based pruning
- Gradient-based pruning
- Lottery ticket hypothesis
- Knowledge distillation
- Quantización de pesos
- Compresión de modelos
- Análisis de importancia de conexiones

Autor: LucIA Development Team
Versión: 4.7.0
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as prune
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
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import math
from dataclasses import dataclass, field
import time
from collections import defaultdict, deque

logger = logging.getLogger('RFENRN1.RF_RFENRN1_4_7')


@dataclass
class PruningConfig:
    """Configuración para técnicas de pruning"""
    pruning_method: str = 'magnitude'  # 'magnitude', 'gradient', 'random', 'lottery_ticket'
    pruning_ratio: float = 0.1
    structured_pruning: bool = False
    gradual_pruning: bool = True
    pruning_steps: int = 10

    # Lottery ticket específico
    lottery_ticket_iterations: int = 3
    lottery_ticket_rewind_epoch: int = 2

    # Knowledge distillation
    use_knowledge_distillation: bool = False
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7

    # Quantización
    use_quantization: bool = False
    quantization_bits: int = 8


class MagnitudePruner:
    """Pruning basado en magnitud de pesos"""

    def __init__(self, config: PruningConfig):
        self.config = config

    def prune_model(self, model: nn.Module, pruning_ratio: float = None) -> nn.Module:
        """Aplica pruning basado en magnitud"""
        pruning_ratio = pruning_ratio or self.config.pruning_ratio

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)

        return model

    def remove_pruning(self, model: nn.Module) -> nn.Module:
        """Remueve el pruning pero mantiene los pesos"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.remove(module, 'weight')

        return model


class GradientPruner:
    """Pruning basado en gradientes"""

    def __init__(self, config: PruningConfig):
        self.config = config

    def prune_model(self, model: nn.Module, pruning_ratio: float = None) -> nn.Module:
        """Aplica pruning basado en gradientes"""
        pruning_ratio = pruning_ratio or self.config.pruning_ratio

        # Calcular importancia basada en gradientes
        importance_scores = self._calculate_gradient_importance(model)

        # Aplicar pruning
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self._prune_by_importance(module, importance_scores.get(name, {}), pruning_ratio)

        return model

    def _calculate_gradient_importance(self, model: nn.Module) -> Dict[str, Dict]:
        """Calcula importancia basada en gradientes"""
        importance_scores = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and module.weight.grad is not None:
                # Importancia = |peso * gradiente|
                importance = torch.abs(module.weight.data * module.weight.grad)
                importance_scores[name] = importance

        return importance_scores

    def _prune_by_importance(self, module: nn.Module, importance: torch.Tensor, pruning_ratio: float):
        """Pruning basado en scores de importancia"""
        if importance.numel() == 0:
            return

        # Calcular threshold
        flat_importance = importance.flatten()
        threshold = torch.quantile(flat_importance, pruning_ratio)

        # Crear máscara
        mask = importance > threshold
        prune.custom_from_mask(module, name='weight', mask=mask)


class LotteryTicketPruner:
    """Implementación de Lottery Ticket Hypothesis"""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.original_weights = {}

    def find_lottery_ticket(self, model: nn.Module, train_fn: Callable,
                            pruning_ratio: float = None) -> nn.Module:
        """Encuentra el lottery ticket"""
        pruning_ratio = pruning_ratio or self.config.pruning_ratio

        # Guardar pesos originales
        self._save_original_weights(model)

        best_model = None
        best_performance = -np.inf

        for iteration in range(self.config.lottery_ticket_iterations):
            # Restaurar pesos originales
            self._restore_original_weights(model)

            # Entrenar brevemente
            performance = train_fn(model, epochs=self.config.lottery_ticket_rewind_epoch)

            # Aplicar pruning
            pruned_model = self._prune_model(model, pruning_ratio)

            # Evaluar rendimiento
            if performance > best_performance:
                best_performance = performance
                best_model = pruned_model

        return best_model

    def _save_original_weights(self, model: nn.Module):
        """Guarda los pesos originales"""
        for name, param in model.named_parameters():
            self.original_weights[name] = param.data.clone()

    def _restore_original_weights(self, model: nn.Module):
        """Restaura los pesos originales"""
        for name, param in model.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name])

    def _prune_model(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """Aplica pruning al modelo"""
        pruner = MagnitudePruner(self.config)
        return pruner.prune_model(model, pruning_ratio)


class KnowledgeDistillation:
    """Knowledge Distillation para compresión de modelos"""

    def __init__(self, config: PruningConfig):
        self.config = config

    def distill_knowledge(self, teacher_model: nn.Module, student_model: nn.Module,
                          train_loader: Any, optimizer: torch.optim.Optimizer) -> nn.Module:
        """Distila conocimiento del teacher al student"""
        teacher_model.eval()
        student_model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass del teacher
            with torch.no_grad():
                teacher_output = teacher_model(data)

            # Forward pass del student
            student_output = student_model(data)

            # Calcular pérdida de distillation
            distillation_loss = self._distillation_loss(student_output, teacher_output, target)

            # Backward pass
            distillation_loss.backward()
            optimizer.step()

        return student_model

    def _distillation_loss(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
                           target: torch.Tensor) -> torch.Tensor:
        """Calcula la pérdida de distillation"""
        # Soft targets del teacher
        teacher_soft = torch.softmax(teacher_output / self.config.distillation_temperature, dim=1)
        student_soft = torch.log_softmax(student_output / self.config.distillation_temperature, dim=1)

        # Distillation loss
        distillation_loss = -torch.sum(teacher_soft * student_soft, dim=1).mean()

        # Hard targets
        hard_loss = nn.CrossEntropyLoss()(student_output, target)

        # Combinar pérdidas
        total_loss = (self.config.distillation_alpha * distillation_loss +
                      (1 - self.config.distillation_alpha) * hard_loss)

        return total_loss


class ModelQuantizer:
    """Quantización de modelos para compresión"""

    def __init__(self, config: PruningConfig):
        self.config = config

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantiza el modelo"""
        if self.config.use_quantization:
            # Quantización dinámica
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            return quantized_model

        return model


class PruningOptimizerManager:
    """
    Gestor de técnicas de pruning y optimización sparse para redes de refuerzo.

    Proporciona un conjunto completo de técnicas de compresión y pruning
    con análisis automático de importancia y optimización de rendimiento.
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Inicializa el gestor de pruning.

        Args:
            config: Configuración de pruning (opcional)
        """
        self.config = config or PruningConfig()
        self.pruner = None
        self.knowledge_distiller = KnowledgeDistillation(self.config) if self.config.use_knowledge_distillation else None
        self.quantizer = ModelQuantizer(self.config)

        self.pruning_history = []
        self.metrics = {
            'total_prunings': 0,
            'compression_ratio': 0.0,
            'accuracy_drop': 0.0,
            'model_size_reduction': 0.0,
            'inference_speedup': 0.0
        }

        logger.info("PruningOptimizerManager inicializado")

    def create_pruner(self) -> Any:
        """Crea el pruner especificado"""
        if self.config.pruning_method == 'magnitude':
            self.pruner = MagnitudePruner(self.config)
        elif self.config.pruning_method == 'gradient':
            self.pruner = GradientPruner(self.config)
        elif self.config.pruning_method == 'lottery_ticket':
            self.pruner = LotteryTicketPruner(self.config)
        else:
            raise ValueError(f"Método de pruning no soportado: {self.config.pruning_method}")

        logger.info(f"Pruner {self.config.pruning_method} creado")
        return self.pruner

    def prune_model(self, model: nn.Module, pruning_ratio: float = None) -> nn.Module:
        """
        Aplica pruning al modelo.

        Args:
            model: Modelo PyTorch
            pruning_ratio: Ratio de pruning (opcional)

        Returns:
            Modelo podado
        """
        if self.pruner is None:
            self.create_pruner()

        start_time = time.time()

        # Medir tamaño original
        original_size = self._calculate_model_size(model)

        # Aplicar pruning
        if self.config.gradual_pruning:
            pruned_model = self._gradual_pruning(model, pruning_ratio)
        else:
            pruned_model = self.pruner.prune_model(model, pruning_ratio)

        # Medir tamaño después del pruning
        pruned_size = self._calculate_model_size(pruned_model)

        # Calcular métricas
        compression_ratio = original_size / pruned_size if pruned_size > 0 else 1.0
        size_reduction = (original_size - pruned_size) / original_size if original_size > 0 else 0.0

        pruning_time = time.time() - start_time

        # Actualizar métricas
        self.metrics['total_prunings'] += 1
        self.metrics['compression_ratio'] = compression_ratio
        self.metrics['model_size_reduction'] = size_reduction

        # Registrar en historial
        pruning_record = {
            'method': self.config.pruning_method,
            'pruning_ratio': pruning_ratio or self.config.pruning_ratio,
            'compression_ratio': compression_ratio,
            'size_reduction': size_reduction,
            'pruning_time': pruning_time,
            'timestamp': time.time()
        }
        self.pruning_history.append(pruning_record)

        logger.info(f"Pruning completado en {pruning_time:.2f}s, compresión: {compression_ratio:.2f}x")

        return pruned_model

    def _gradual_pruning(self, model: nn.Module, final_ratio: float) -> nn.Module:
        """Pruning gradual para mejor rendimiento"""
        current_ratio = 0.0
        step_ratio = final_ratio / self.config.pruning_steps

        for step in range(self.config.pruning_steps):
            current_ratio += step_ratio
            model = self.pruner.prune_model(model, current_ratio)

        return model

    def _calculate_model_size(self, model: nn.Module) -> int:
        """Calcula el tamaño del modelo en parámetros"""
        return sum(p.numel() for p in model.parameters())

    def compress_model(self, model: nn.Module, teacher_model: nn.Module = None) -> nn.Module:
        """
        Aplica compresión completa al modelo.

        Args:
            model: Modelo a comprimir
            teacher_model: Modelo teacher para distillation (opcional)

        Returns:
            Modelo comprimido
        """
        compressed_model = model

        # Knowledge distillation si está disponible
        if self.knowledge_distiller and teacher_model is not None:
            logger.info("Aplicando knowledge distillation...")
            # En implementación real, esto requeriría un dataloader
            # compressed_model = self.knowledge_distiller.distill_knowledge(teacher_model, compressed_model, train_loader, optimizer)

        # Pruning
        compressed_model = self.prune_model(compressed_model)

        # Quantización
        compressed_model = self.quantizer.quantize_model(compressed_model)

        logger.info("Compresión completa aplicada")

        return compressed_model

    def analyze_model_sparsity(self, model: nn.Module) -> Dict[str, Any]:
        """Analiza la sparsity del modelo"""
        sparsity_analysis = {
            'total_parameters': 0,
            'zero_parameters': 0,
            'sparsity_ratio': 0.0,
            'layer_sparsity': {}
        }

        total_params = 0
        zero_params = 0

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                layer_params = weight.numel()
                layer_zeros = (weight == 0).sum().item()

                sparsity_analysis['layer_sparsity'][name] = {
                    'total': layer_params,
                    'zeros': layer_zeros,
                    'sparsity': layer_zeros / layer_params if layer_params > 0 else 0.0
                }

                total_params += layer_params
                zero_params += layer_zeros

        sparsity_analysis['total_parameters'] = total_params
        sparsity_analysis['zero_parameters'] = zero_params
        sparsity_analysis['sparsity_ratio'] = zero_params / total_params if total_params > 0 else 0.0

        return sparsity_analysis

    def get_compression_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de la compresión"""
        return {
            'config': self.config,
            'metrics': self.metrics.copy(),
            'total_prunings': self.metrics['total_prunings'],
            'average_compression_ratio': self.metrics['compression_ratio'],
            'average_size_reduction': self.metrics['model_size_reduction'],
            'history_length': len(self.pruning_history)
        }

    def save_state(self, path: str) -> None:
        """Guarda el estado del pruner"""
        torch.save({
            'config': self.config,
            'metrics': self.metrics,
            'pruning_history': self.pruning_history,
            'original_weights': getattr(self.pruner, 'original_weights', {}) if self.pruner else {}
        }, path)
        logger.info(f"Estado del pruner guardado en {path}")

    def load_state(self, path: str) -> None:
        """Carga el estado del pruner"""
        checkpoint = torch.load(path)

        self.config = checkpoint.get('config', self.config)
        self.metrics = checkpoint.get('metrics', self.metrics)
        self.pruning_history = checkpoint.get('pruning_history', [])

        # Restaurar pesos originales si es lottery ticket
        if self.pruner and 'original_weights' in checkpoint:
            self.pruner.original_weights = checkpoint['original_weights']

        logger.info(f"Estado del pruner cargado desde {path}")

    def reset_state(self) -> None:
        """Reinicia el estado del pruner"""
        self.metrics = {
            'total_prunings': 0,
            'compression_ratio': 0.0,
            'accuracy_drop': 0.0,
            'model_size_reduction': 0.0,
            'inference_speedup': 0.0
        }

        self.pruning_history.clear()

        if self.pruner:
            self.pruner.original_weights = {}

        logger.info("Estado del pruner reiniciado")
