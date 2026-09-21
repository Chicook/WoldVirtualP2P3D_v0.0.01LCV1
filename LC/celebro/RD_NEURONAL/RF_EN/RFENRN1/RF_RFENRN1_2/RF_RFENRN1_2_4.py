"""
RF_RFENRN1_2_4.py - Gestor de Uso Eficiente de Memoria y Cómputo
================================================================

Implementa técnicas avanzadas para optimizar el uso de memoria y cómputo
en el entrenamiento de redes neuronales de aprendizaje por refuerzo.
Incluye Mixed Precision, Activation Checkpointing, ZeRO, LoRA y técnicas
de cuantización para reducir el uso de recursos.

Características:
- Mixed Precision Training (FP16/AMP) con loss scaling automático
- Activation Checkpointing para ahorrar memoria
- Optimizer State Sharding (ZeRO) para modelos grandes
- Quantization-Aware Training (QAT) para inferencia eficiente
- LoRA y Adapters para fine-tuning eficiente en parámetros

Autor: LucIA Development Team
Versión: 2.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import gc
import psutil
import os
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger('RFENRN1.RF_RFENRN1_2_4')


@dataclass
class MemoryOptimizationConfig:
    """Configuración para optimización de memoria"""
    use_mixed_precision: bool = True
    use_activation_checkpointing: bool = True
    checkpoint_segments: int = 4
    use_gradient_checkpointing: bool = True
    use_zero_optimizer: bool = False
    zero_stage: int = 1
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    use_quantization: bool = False
    quantization_bits: int = 8
    memory_efficient_attention: bool = True
    gradient_accumulation_steps: int = 1
    cpu_offload: bool = False


class MemoryOptimizationManager:
    """
    Gestor de optimización de memoria y cómputo.

    Implementa técnicas avanzadas para reducir el uso de memoria
    y acelerar el entrenamiento en redes de refuerzo.
    """

    def __init__(self, config: Optional[MemoryOptimizationConfig] = None):
        """
        Inicializa el gestor de optimización de memoria.

        Args:
            config: Configuración de optimización de memoria (opcional)
        """
        self.config = config or MemoryOptimizationConfig()
        # Activar GradScaler solo si hay CUDA para evitar warnings en CPU
        if self.config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
            # Si no hay CUDA, desactivar mixed precision para evitar ruido en logs
            if not torch.cuda.is_available():
                self.config.use_mixed_precision = False
        self.memory_stats = {
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
            'memory_saved_by_checkpointing': 0.0,
            'memory_saved_by_mixed_precision': 0.0,
            'optimization_efficiency': 0.0
        }
        self.checkpoint_counter = 0

        logger.info("MemoryOptimizationManager inicializado")

    @contextmanager
    def mixed_precision_context(self):
        """
        Context manager para Mixed Precision Training.

        Yields:
            Contexto de precisión mixta
        """
        if not self.config.use_mixed_precision:
            yield
            return

        with amp.autocast():
            yield

    def apply_activation_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        Aplica Activation Checkpointing al modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con checkpointing aplicado
        """
        if not self.config.use_activation_checkpointing:
            return model

        # Aplicar checkpointing a capas específicas
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                # Wrapper para checkpointing
                original_forward = module.forward

                def checkpointed_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(
                        original_forward, *args, **kwargs,
                        use_reentrant=False
                    )

                module.forward = checkpointed_forward

        logger.info("Activation Checkpointing aplicado al modelo")
        return model

    def apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        Aplica Gradient Checkpointing al modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con gradient checkpointing aplicado
        """
        if not self.config.use_gradient_checkpointing:
            return model

        # Aplicar gradient checkpointing
        torch.utils.checkpoint.checkpoint_sequential(
            model,
            self.config.checkpoint_segments,
            None
        )

        logger.info("Gradient Checkpointing aplicado al modelo")
        return model

    def setup_lora_adapters(self, model: nn.Module) -> nn.Module:
        """
        Configura LoRA adapters para fine-tuning eficiente.

        Args:
            model: Modelo PyTorch

        Returns:
            Modelo con LoRA adapters
        """
        if not self.config.use_lora:
            return model

        try:
            # Implementación básica de LoRA
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Crear adapters LoRA
                    lora_adapter = LoRAAdapter(
                        module.in_features,
                        module.out_features,
                        rank=self.config.lora_rank,
                        alpha=self.config.lora_alpha
                    )
                    # Reemplazar módulo original
                    setattr(model, name, lora_adapter)

            logger.info("LoRA adapters configurados")
        except Exception as e:
            logger.warning(f"Error configurando LoRA: {e}")

        return model

    def optimize_memory_usage(self, model: nn.Module) -> Dict[str, float]:
        """
        Optimiza el uso de memoria del modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Métricas de optimización de memoria
        """
        initial_memory = self.get_memory_usage()

        # Aplicar optimizaciones
        if self.config.use_mixed_precision:
            model = model.half()
            self.memory_stats['memory_saved_by_mixed_precision'] = initial_memory - self.get_memory_usage()

        if self.config.use_activation_checkpointing:
            model = self.apply_activation_checkpointing(model)

        if self.config.cpu_offload:
            # Offload de parámetros a CPU
            for param in model.parameters():
                param.data = param.data.cpu()

        final_memory = self.get_memory_usage()
        self.memory_stats['current_memory_usage'] = final_memory
        self.memory_stats['peak_memory_usage'] = max(
            self.memory_stats['peak_memory_usage'],
            final_memory
        )

        return self.memory_stats.copy()

    def get_memory_usage(self) -> float:
        """
        Obtiene el uso actual de memoria en MB.

        Returns:
            Uso de memoria en MB
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024

    def clear_memory_cache(self) -> None:
        """Limpia la caché de memoria."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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

        try:
            # Cuantización dinámica
            if self.config.quantization_bits == 8:
                model = torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )
            elif self.config.quantization_bits == 16:
                model = model.half()

            logger.info(f"Modelo cuantizado a {self.config.quantization_bits} bits")
        except Exception as e:
            logger.warning(f"Error en cuantización: {e}")

        return model

    def setup_gradient_accumulation(self, optimizer, accumulation_steps: int) -> None:
        """
        Configura gradient accumulation.

        Args:
            optimizer: Optimizador PyTorch
            accumulation_steps: Pasos de acumulación
        """
        self.config.gradient_accumulation_steps = accumulation_steps
        logger.info(f"Gradient accumulation configurado para {accumulation_steps} pasos")

    def calculate_optimization_efficiency(self) -> float:
        """
        Calcula la eficiencia de optimización de memoria.

        Returns:
            Score de eficiencia (0-1, mayor es mejor)
        """
        if self.memory_stats['peak_memory_usage'] == 0:
            return 0.0

        # Calcular eficiencia basada en memoria ahorrada
        total_saved = (
            self.memory_stats['memory_saved_by_checkpointing'] +
            self.memory_stats['memory_saved_by_mixed_precision']
        )

        efficiency = min(1.0, total_saved / self.memory_stats['peak_memory_usage'])
        self.memory_stats['optimization_efficiency'] = efficiency

        return efficiency

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Obtiene las estadísticas de memoria.

        Returns:
            Diccionario con estadísticas de memoria
        """
        return self.memory_stats.copy()

    def reset_memory_stats(self) -> None:
        """Reinicia las estadísticas de memoria."""
        self.memory_stats = {
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
            'memory_saved_by_checkpointing': 0.0,
            'memory_saved_by_mixed_precision': 0.0,
            'optimization_efficiency': 0.0
        }
        logger.info("Estadísticas de memoria reiniciadas")


class LoRAAdapter(nn.Module):
    """
    Adapter LoRA para fine-tuning eficiente.

    Implementa Low-Rank Adaptation para reducir parámetros
    durante el fine-tuning de modelos grandes.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0):
        """
        Inicializa el adapter LoRA.

        Args:
            in_features: Características de entrada
            out_features: Características de salida
            rank: Rango de la descomposición
            alpha: Factor de escalado
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Matrices de baja dimensión
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Matriz original (congelada)
        self.original_weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del adapter LoRA.

        Args:
            x: Tensor de entrada

        Returns:
            Tensor de salida
        """
        # Calcular adaptación LoRA
        lora_weight = self.lora_B @ self.lora_A
        adapted_weight = self.original_weight + (self.alpha / self.rank) * lora_weight

        return F.linear(x, adapted_weight)
