"""
RF_RFENRN1_2_9.py - Gestor de Técnicas de Sparsity y Compresión
===============================================================

Implementa técnicas avanzadas de sparsity y compresión para redes neuronales
de aprendizaje por refuerzo. Incluye Pruning estructurado y no estructurado,
Dynamic Sparse Training, compresión de modelos y técnicas de cuantización
para reducir el tamaño y acelerar la inferencia.

Características:
- Pruning estructurado y no estructurado con criterios adaptativos
- Dynamic Sparse Training (DST) para mantener sparsity durante entrenamiento
- Compresión de modelos con técnicas avanzadas
- Cuantización post-entrenamiento e intrainment
- Sparse matrices y kernels optimizados
- Técnicas de knowledge distillation para modelos compactos

Autor: LucIA Development Team
Versión: 2.0.0
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as prune
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
from dataclasses import dataclass
import copy
from abc import ABC, abstractmethod

logger = logging.getLogger('RFENRN1.RF_RFENRN1_2_9')


@dataclass
class SparsityConfig:
    """Configuración para técnicas de sparsity"""
    use_pruning: bool = True
    pruning_type: str = 'unstructured'  # 'unstructured', 'structured', 'global'
    pruning_ratio: float = 0.1
    pruning_criterion: str = 'magnitude'  # 'magnitude', 'gradient', 'random'
    use_dynamic_sparse_training: bool = True
    dst_sparsity_ratio: float = 0.5
    dst_growth_ratio: float = 0.1
    dst_regrowth_frequency: int = 100
    use_quantization: bool = True
    quantization_bits: int = 8
    quantization_type: str = 'post_training'  # 'post_training', 'qat'
    use_knowledge_distillation: bool = True
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7
    compression_target: float = 0.5  # Reducir modelo a 50% del tamaño original


class SparsityManager:
    """
    Gestor de técnicas de sparsity y compresión.

    Implementa técnicas avanzadas para crear modelos sparse y comprimidos
    manteniendo el rendimiento en redes de aprendizaje por refuerzo.
    """

    def __init__(self, config: Optional[SparsityConfig] = None):
        """
        Inicializa el gestor de sparsity.

        Args:
            config: Configuración de sparsity (opcional)
        """
        self.config = config or SparsityConfig()
        self.pruned_modules = {}
        self.sparse_masks = {}
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 1.0,
            'sparsity_ratio': 0.0,
            'accuracy_drop': 0.0,
            'speedup_factor': 1.0
        }
        self.dst_step_counter = 0

        logger.info("SparsityManager inicializado")

    def apply_pruning(self, model: nn.Module, pruning_ratio: float = None) -> nn.Module:
        """
        Aplica pruning al modelo.

        Args:
            model: Modelo PyTorch
            pruning_ratio: Ratio de pruning (opcional)

        Returns:
            Modelo con pruning aplicado
        """
        if not self.config.use_pruning:
            return model

        ratio = pruning_ratio or self.config.pruning_ratio

        if self.config.pruning_type == 'unstructured':
            return self._apply_unstructured_pruning(model, ratio)
        elif self.config.pruning_type == 'structured':
            return self._apply_structured_pruning(model, ratio)
        elif self.config.pruning_type == 'global':
            return self._apply_global_pruning(model, ratio)
        else:
            logger.warning(f"Tipo de pruning no soportado: {self.config.pruning_type}")
            return model

    def _apply_unstructured_pruning(self, model: nn.Module, ratio: float) -> nn.Module:
        """
        Aplica pruning no estructurado.

        Args:
            model: Modelo PyTorch
            ratio: Ratio de pruning

        Returns:
            Modelo con pruning no estructurado
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                if self.config.pruning_criterion == 'magnitude':
                    prune.l1_unstructured(module, name='weight', amount=ratio)
                elif self.config.pruning_criterion == 'random':
                    prune.random_unstructured(module, name='weight', amount=ratio)

                self.pruned_modules[name] = module
                logger.debug(f"Pruning no estructurado aplicado a {name}")

        logger.info(f"Pruning no estructurado aplicado con ratio {ratio}")
        return model

    def _apply_structured_pruning(self, model: nn.Module, ratio: float) -> nn.Module:
        """
        Aplica pruning estructurado.

        Args:
            model: Modelo PyTorch
            ratio: Ratio de pruning

        Returns:
            Modelo con pruning estructurado
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(module, name='weight', amount=ratio, n=2, dim=0)
            elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
                prune.ln_structured(module, name='weight', amount=ratio, n=2, dim=0)

            self.pruned_modules[name] = module
            logger.debug(f"Pruning estructurado aplicado a {name}")

        logger.info(f"Pruning estructurado aplicado con ratio {ratio}")
        return model

    def _apply_global_pruning(self, model: nn.Module, ratio: float) -> nn.Module:
        """
        Aplica pruning global.

        Args:
            model: Modelo PyTorch
            ratio: Ratio de pruning

        Returns:
            Modelo con pruning global
        """
        parameters_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=ratio
        )

        logger.info(f"Pruning global aplicado con ratio {ratio}")
        return model

    def setup_dynamic_sparse_training(self, model: nn.Module) -> nn.Module:
        """
        Configura Dynamic Sparse Training.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con DST configurado
        """
        if not self.config.use_dynamic_sparse_training:
            return model

        # Crear máscaras sparse iniciales
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                mask = self._create_sparse_mask(module.weight, self.config.dst_sparsity_ratio)
                self.sparse_masks[name] = mask

                # Aplicar máscara inicial
                module.weight.data *= mask

        logger.info("Dynamic Sparse Training configurado")
        return model

    def _create_sparse_mask(self, weight: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
        """
        Crea una máscara sparse para los pesos.

        Args:
            weight: Tensor de pesos
            sparsity_ratio: Ratio de sparsity

        Returns:
            Máscara sparse
        """
        mask = torch.ones_like(weight)
        num_params = weight.numel()
        num_pruned = int(num_params * sparsity_ratio)

        # Seleccionar parámetros a podar basándose en magnitud
        flat_weight = weight.flatten()
        _, indices = torch.topk(torch.abs(flat_weight), num_pruned, largest=False)

        flat_mask = mask.flatten()
        flat_mask[indices] = 0

        return flat_mask.reshape(weight.shape)

    def update_dynamic_sparsity(self, model: nn.Module, step: int) -> None:
        """
        Actualiza la sparsity dinámicamente durante el entrenamiento.

        Args:
            model: Modelo PyTorch
            step: Paso actual de entrenamiento
        """
        if not self.config.use_dynamic_sparse_training:
            return

        self.dst_step_counter += 1

        if self.dst_step_counter % self.config.dst_regrowth_frequency == 0:
            self._regrow_connections(model)

    def _regrow_connections(self, model: nn.Module) -> None:
        """
        Regenera conexiones basándose en gradientes.

        Args:
            model: Modelo PyTorch
        """
        for name, module in model.named_modules():
            if name in self.sparse_masks and isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                if module.weight.grad is not None:
                    # Calcular gradientes de parámetros podados
                    mask = self.sparse_masks[name]
                    pruned_gradients = module.weight.grad * (1 - mask)

                    # Encontrar parámetros con mayor gradiente para regenerar
                    flat_grad = pruned_gradients.flatten()
                    num_regrow = int(flat_grad.numel() * self.config.dst_growth_ratio)

                    if num_regrow > 0:
                        _, regrow_indices = torch.topk(torch.abs(flat_grad), num_regrow)

                        # Actualizar máscara
                        flat_mask = mask.flatten()
                        flat_mask[regrow_indices] = 1
                        self.sparse_masks[name] = flat_mask.reshape(mask.shape)

                        # Regenerar parámetros
                        module.weight.data[regrow_indices] = torch.randn_like(
                            module.weight.data[regrow_indices]
                        ) * 0.01

    def apply_quantization(self, model: nn.Module) -> nn.Module:
        """
        Aplica cuantización al modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo cuantizado
        """
        if not self.config.use_quantization:
            return model

        if self.config.quantization_type == 'post_training':
            return self._apply_post_training_quantization(model)
        elif self.config.quantization_type == 'qat':
            return self._apply_quantization_aware_training(model)
        else:
            logger.warning(f"Tipo de cuantización no soportado: {self.config.quantization_type}")
            return model

    def _apply_post_training_quantization(self, model: nn.Module) -> nn.Module:
        """
        Aplica cuantización post-entrenamiento.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo cuantizado
        """
        if self.config.quantization_bits == 8:
            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
        elif self.config.quantization_bits == 16:
            model = model.half()

        logger.info(f"Cuantización post-entrenamiento aplicada ({self.config.quantization_bits} bits)")
        return model

    def _apply_quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """
        Aplica cuantización durante el entrenamiento.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con QAT aplicado
        """
        # Preparar modelo para QAT
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model = torch.quantization.prepare_qat(model)

        logger.info("Quantization-Aware Training configurado")
        return model

    def apply_knowledge_distillation(self, teacher_model: nn.Module,
                                     student_model: nn.Module) -> nn.Module:
        """
        Aplica knowledge distillation para crear un modelo compacto.

        Args:
            teacher_model: Modelo maestro
            student_model: Modelo estudiante

        Returns:
            Modelo estudiante optimizado
        """
        if not self.config.use_knowledge_distillation:
            return student_model

        # Crear wrapper para distillation
        distillation_wrapper = KnowledgeDistillationWrapper(
            teacher_model,
            student_model,
            temperature=self.config.distillation_temperature,
            alpha=self.config.distillation_alpha
        )

        logger.info("Knowledge Distillation configurado")
        return distillation_wrapper

    def calculate_model_size(self, model: nn.Module) -> Dict[str, int]:
        """
        Calcula el tamaño del modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Diccionario con estadísticas de tamaño
        """
        total_params = 0
        trainable_params = 0

        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        # Calcular tamaño en bytes
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'size_bytes': total_size,
            'size_mb': total_size / (1024 * 1024)
        }

    def calculate_sparsity_ratio(self, model: nn.Module) -> float:
        """
        Calcula el ratio de sparsity del modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Ratio de sparsity (0-1)
        """
        total_params = 0
        zero_params = 0

        for name, param in model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

        if total_params == 0:
            return 0.0

        sparsity_ratio = zero_params / total_params
        self.compression_stats['sparsity_ratio'] = sparsity_ratio

        return sparsity_ratio

    def compress_model(self, model: nn.Module) -> nn.Module:
        """
        Comprime el modelo usando técnicas combinadas.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo comprimido
        """
        original_size = self.calculate_model_size(model)
        self.compression_stats['original_size'] = original_size['size_mb']

        # Aplicar técnicas de compresión
        compressed_model = copy.deepcopy(model)

        # 1. Pruning
        compressed_model = self.apply_pruning(compressed_model)

        # 2. Cuantización
        compressed_model = self.apply_quantization(compressed_model)

        # 3. Calcular estadísticas finales
        compressed_size = self.calculate_model_size(compressed_model)
        self.compression_stats['compressed_size'] = compressed_size['size_mb']

        # Calcular ratio de compresión
        if self.compression_stats['original_size'] > 0:
            compression_ratio = (
                self.compression_stats['compressed_size'] /
                self.compression_stats['original_size']
            )
            self.compression_stats['compression_ratio'] = compression_ratio

        logger.info(f"Modelo comprimido: {self.compression_stats['original_size']:.2f}MB -> "
                    f"{self.compression_stats['compressed_size']:.2f}MB "
                    f"(ratio: {self.compression_stats['compression_ratio']:.2f})")

        return compressed_model

    def get_compression_stats(self) -> Dict[str, float]:
        """
        Obtiene las estadísticas de compresión.

        Returns:
            Diccionario con estadísticas
        """
        return self.compression_stats.copy()

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de compresión."""
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 1.0,
            'sparsity_ratio': 0.0,
            'accuracy_drop': 0.0,
            'speedup_factor': 1.0
        }

        logger.info("Estadísticas de compresión reiniciadas")


class KnowledgeDistillationWrapper(nn.Module):
    """
    Wrapper para Knowledge Distillation.

    Implementa la técnica de distillation para transferir conocimiento
    de un modelo maestro a un modelo estudiante más compacto.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module,
                 temperature: float = 3.0, alpha: float = 0.7):
        """
        Inicializa el wrapper de distillation.

        Args:
            teacher: Modelo maestro
            student: Modelo estudiante
            temperature: Temperatura para softmax
            alpha: Peso para loss de distillation
        """
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        # Congelar modelo maestro
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con distillation.

        Args:
            x: Tensor de entrada

        Returns:
            Salida del modelo estudiante
        """
        return self.student(x)

    def distillation_loss(self, student_outputs: torch.Tensor,
                          teacher_outputs: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida de distillation.

        Args:
            student_outputs: Salidas del estudiante
            teacher_outputs: Salidas del maestro
            targets: Targets reales

        Returns:
            Pérdida de distillation
        """
        # Softmax con temperatura
        student_soft = F.softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)

        # Loss de distillation (KL divergence)
        distillation_loss = F.kl_div(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Loss de clasificación normal
        classification_loss = F.cross_entropy(student_outputs, targets)

        # Combinar losses
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * classification_loss

        return total_loss
