"""
RF_RFENRN1_2_8.py - Gestor de Reducción de Comunicación Distribuida
===================================================================

Implementa técnicas avanzadas para reducir el overhead de comunicación
en entrenamiento distribuido de redes neuronales de aprendizaje por refuerzo.
Incluye Gradient Compression, Quantized Gradients, comunicación asíncrona
y técnicas de compresión para optimizar el rendimiento en clusters.

Características:
- Gradient Compression (top-k sparsity, random sparsity)
- Quantized Gradients (FP16, INT8, INT4) para comunicación
- Comunicación asíncrona con buffers adaptativos
- Compresión de gradientes con algoritmos eficientes
- Reducción de frecuencia de comunicación
- Técnicas de agregación inteligente de gradientes

Autor: LucIA Development Team
Versión: 2.0.0
"""

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
import random
from dataclasses import dataclass
import threading
import queue
from collections import defaultdict

logger = logging.getLogger('RFENRN1.RF_RFENRN1_2_8')


@dataclass
class CommunicationConfig:
    """Configuración para reducción de comunicación"""
    use_gradient_compression: bool = True
    compression_ratio: float = 0.1  # Mantener top 10% de gradientes
    compression_type: str = 'topk'  # 'topk', 'random', 'threshold'
    use_quantization: bool = True
    quantization_bits: int = 16  # 16, 8, 4
    use_async_communication: bool = True
    communication_frequency: int = 1  # Cada N pasos
    buffer_size: int = 1000
    use_error_feedback: bool = True
    error_feedback_momentum: float = 0.9
    use_gradient_accumulation: bool = True
    accumulation_steps: int = 4
    compression_threshold: float = 1e-6


class CommunicationOptimizer:
    """
    Optimizador de comunicación distribuida.

    Implementa técnicas avanzadas para reducir el overhead de comunicación
    en entrenamiento distribuido de redes de aprendizaje por refuerzo.
    """

    def __init__(self, config: Optional[CommunicationConfig] = None):
        """
        Inicializa el optimizador de comunicación.

        Args:
            config: Configuración de comunicación (opcional)
        """
        self.config = config or CommunicationConfig()
        self.compression_buffer = {}
        self.error_feedback = {}
        self.communication_stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'compression_ratio': 1.0,
            'communication_time': 0.0,
            'bandwidth_utilization': 0.0,
            'compression_efficiency': 0.0
        }
        self.gradient_accumulator = {}
        self.step_counter = 0

        logger.info("CommunicationOptimizer inicializado")

    def compress_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Comprime gradientes usando técnicas avanzadas.

        Args:
            gradients: Diccionario de gradientes por nombre

        Returns:
            Gradientes comprimidos
        """
        if not self.config.use_gradient_compression:
            return gradients

        compressed_gradients = {}

        for name, grad in gradients.items():
            if grad is None:
                continue

            if self.config.compression_type == 'topk':
                compressed = self._compress_topk(grad, name)
            elif self.config.compression_type == 'random':
                compressed = self._compress_random(grad, name)
            elif self.config.compression_type == 'threshold':
                compressed = self._compress_threshold(grad, name)
            else:
                compressed = grad

            compressed_gradients[name] = compressed

        return compressed_gradients

    def _compress_topk(self, grad: torch.Tensor, name: str) -> Dict[str, Any]:
        """
        Comprime gradientes usando top-k sparsity.

        Args:
            grad: Gradiente a comprimir
            name: Nombre del parámetro

        Returns:
            Gradiente comprimido
        """
        # Calcular número de elementos a mantener
        total_elements = grad.numel()
        k = max(1, int(total_elements * self.config.compression_ratio))

        # Encontrar top-k elementos
        flat_grad = grad.flatten()
        _, topk_indices = torch.topk(torch.abs(flat_grad), k)

        # Crear gradiente comprimido
        compressed_values = flat_grad[topk_indices]

        # Aplicar error feedback si está habilitado
        if self.config.use_error_feedback and name in self.error_feedback:
            error = self.error_feedback[name]
            compressed_values = compressed_values + error[topk_indices]

        return {
            'values': compressed_values,
            'indices': topk_indices,
            'shape': grad.shape,
            'compression_type': 'topk'
        }

    def _compress_random(self, grad: torch.Tensor, name: str) -> Dict[str, Any]:
        """
        Comprime gradientes usando random sparsity.

        Args:
            grad: Gradiente a comprimir
            name: Nombre del parámetro

        Returns:
            Gradiente comprimido
        """
        total_elements = grad.numel()
        k = max(1, int(total_elements * self.config.compression_ratio))

        # Selección aleatoria de elementos
        random_indices = torch.randperm(total_elements)[:k]
        flat_grad = grad.flatten()

        compressed_values = flat_grad[random_indices]

        return {
            'values': compressed_values,
            'indices': random_indices,
            'shape': grad.shape,
            'compression_type': 'random'
        }

    def _compress_threshold(self, grad: torch.Tensor, name: str) -> Dict[str, Any]:
        """
        Comprime gradientes usando threshold.

        Args:
            grad: Gradiente a comprimir
            name: Nombre del parámetro

        Returns:
            Gradiente comprimido
        """
        threshold = self.config.compression_threshold

        # Crear máscara basada en threshold
        mask = torch.abs(grad) > threshold
        indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]

        if len(indices) == 0:
            # Si no hay elementos significativos, mantener el más grande
            flat_grad = grad.flatten()
            max_idx = torch.argmax(torch.abs(flat_grad))
            indices = torch.tensor([max_idx])

        flat_grad = grad.flatten()
        compressed_values = flat_grad[indices]

        return {
            'values': compressed_values,
            'indices': indices,
            'shape': grad.shape,
            'compression_type': 'threshold'
        }

    def decompress_gradients(self, compressed_gradients: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Descomprime gradientes.

        Args:
            compressed_gradients: Gradientes comprimidos

        Returns:
            Gradientes descomprimidos
        """
        decompressed = {}

        for name, compressed in compressed_gradients.items():
            if isinstance(compressed, dict):
                # Reconstruir tensor desde compresión
                zeros = torch.zeros(compressed['shape'], device=compressed['values'].device)
                flat_zeros = zeros.flatten()
                flat_zeros[compressed['indices']] = compressed['values']
                decompressed[name] = flat_zeros.reshape(compressed['shape'])
            else:
                decompressed[name] = compressed

        return decompressed

    def quantize_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Cuantiza gradientes para comunicación eficiente.

        Args:
            gradients: Gradientes a cuantizar

        Returns:
            Gradientes cuantizados
        """
        if not self.config.use_quantization:
            return gradients

        quantized = {}

        for name, grad in gradients.items():
            if grad is None:
                continue

            if self.config.quantization_bits == 16:
                quantized[name] = self._quantize_fp16(grad)
            elif self.config.quantization_bits == 8:
                quantized[name] = self._quantize_int8(grad)
            elif self.config.quantization_bits == 4:
                quantized[name] = self._quantize_int4(grad)
            else:
                quantized[name] = grad

        return quantized

    def _quantize_fp16(self, grad: torch.Tensor) -> Dict[str, Any]:
        """
        Cuantiza gradiente a FP16.

        Args:
            grad: Gradiente a cuantizar

        Returns:
            Gradiente cuantizado
        """
        return {
            'data': grad.half(),
            'scale': torch.tensor(1.0),
            'quantization_type': 'fp16'
        }

    def _quantize_int8(self, grad: torch.Tensor) -> Dict[str, Any]:
        """
        Cuantiza gradiente a INT8.

        Args:
            grad: Gradiente a cuantizar

        Returns:
            Gradiente cuantizado
        """
        # Calcular escala para cuantización
        max_val = torch.max(torch.abs(grad))
        scale = max_val / 127.0 if max_val > 0 else torch.tensor(1.0)

        # Cuantizar
        quantized = torch.round(grad / scale).clamp(-128, 127).to(torch.int8)

        return {
            'data': quantized,
            'scale': scale,
            'quantization_type': 'int8'
        }

    def _quantize_int4(self, grad: torch.Tensor) -> Dict[str, Any]:
        """
        Cuantiza gradiente a INT4.

        Args:
            grad: Gradiente a cuantizar

        Returns:
            Gradiente cuantizado
        """
        # Calcular escala para cuantización INT4
        max_val = torch.max(torch.abs(grad))
        scale = max_val / 7.0 if max_val > 0 else torch.tensor(1.0)

        # Cuantizar a INT4 (usando INT8 como contenedor)
        quantized = torch.round(grad / scale).clamp(-8, 7).to(torch.int8)

        return {
            'data': quantized,
            'scale': scale,
            'quantization_type': 'int4'
        }

    def dequantize_gradients(self, quantized_gradients: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Descuantiza gradientes.

        Args:
            quantized_gradients: Gradientes cuantizados

        Returns:
            Gradientes descuantizados
        """
        dequantized = {}

        for name, quantized in quantized_gradients.items():
            if isinstance(quantized, dict):
                if quantized['quantization_type'] == 'fp16':
                    dequantized[name] = quantized['data'].float()
                elif quantized['quantization_type'] in ['int8', 'int4']:
                    dequantized[name] = quantized['data'].float() * quantized['scale']
                else:
                    dequantized[name] = quantized['data']
            else:
                dequantized[name] = quantized

        return dequantized

    def accumulate_gradients(self, gradients: Dict[str, torch.Tensor]) -> None:
        """
        Acumula gradientes para comunicación menos frecuente.

        Args:
            gradients: Gradientes a acumular
        """
        if not self.config.use_gradient_accumulation:
            return

        for name, grad in gradients.items():
            if name not in self.gradient_accumulator:
                self.gradient_accumulator[name] = torch.zeros_like(grad)

            self.gradient_accumulator[name] += grad

    def should_communicate(self) -> bool:
        """
        Determina si debe comunicar gradientes en este paso.

        Returns:
            True si debe comunicar
        """
        self.step_counter += 1
        return self.step_counter % self.config.communication_frequency == 0

    def get_accumulated_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Obtiene gradientes acumulados y los reinicia.

        Returns:
            Gradientes acumulados
        """
        accumulated = self.gradient_accumulator.copy()

        # Reiniciar acumulador
        self.gradient_accumulator.clear()

        return accumulated

    def update_error_feedback(self, original_grad: torch.Tensor,
                              compressed_grad: torch.Tensor, name: str) -> None:
        """
        Actualiza error feedback para compresión.

        Args:
            original_grad: Gradiente original
            compressed_grad: Gradiente comprimido
            name: Nombre del parámetro
        """
        if not self.config.use_error_feedback:
            return

        # Calcular error de compresión
        error = original_grad - compressed_grad

        if name not in self.error_feedback:
            self.error_feedback[name] = torch.zeros_like(original_grad)

        # Actualizar con momentum
        self.error_feedback[name] = (
            self.config.error_feedback_momentum * self.error_feedback[name] + error
        )

    def communicate_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Comunica gradientes usando técnicas optimizadas.

        Args:
            gradients: Gradientes a comunicar

        Returns:
            Gradientes agregados
        """
        if not dist.is_initialized():
            return gradients

        start_time = time.time()

        # Comprimir gradientes
        compressed = self.compress_gradients(gradients)

        # Cuantizar gradientes
        quantized = self.quantize_gradients(compressed)

        # Comunicar gradientes cuantizados
        aggregated = {}

        for name, quantized_grad in quantized.items():
            if isinstance(quantized_grad, dict):
                # Comunicar datos cuantizados
                dist.all_reduce(quantized_grad['data'])
                aggregated[name] = quantized_grad
            else:
                # Comunicar gradiente normal
                dist.all_reduce(quantized_grad)
                aggregated[name] = quantized_grad

        # Descuantizar gradientes agregados
        dequantized = self.dequantize_gradients(aggregated)

        # Descomprimir gradientes
        final_gradients = self.decompress_gradients(dequantized)

        # Normalizar por número de procesos
        world_size = dist.get_world_size()
        for name in final_gradients:
            final_gradients[name] /= world_size

        # Actualizar estadísticas
        communication_time = time.time() - start_time
        self.communication_stats['communication_time'] += communication_time

        # Calcular bytes comunicados (estimación)
        total_bytes = sum(grad.numel() * grad.element_size() for grad in final_gradients.values())
        self.communication_stats['bytes_sent'] += total_bytes
        self.communication_stats['bytes_received'] += total_bytes

        return final_gradients

    def calculate_compression_efficiency(self) -> float:
        """
        Calcula la eficiencia de compresión.

        Returns:
            Eficiencia de compresión (0-1, mayor es mejor)
        """
        if self.communication_stats['bytes_sent'] == 0:
            return 0.0

        # Calcular ratio de compresión
        original_size = self.communication_stats['bytes_sent'] / self.config.compression_ratio
        compressed_size = self.communication_stats['bytes_sent']

        efficiency = 1.0 - (compressed_size / original_size)
        self.communication_stats['compression_efficiency'] = efficiency

        return efficiency

    def get_communication_stats(self) -> Dict[str, float]:
        """
        Obtiene las estadísticas de comunicación.

        Returns:
            Diccionario con estadísticas
        """
        return self.communication_stats.copy()

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de comunicación."""
        self.communication_stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'compression_ratio': 1.0,
            'communication_time': 0.0,
            'bandwidth_utilization': 0.0,
            'compression_efficiency': 0.0
        }
        self.step_counter = 0

        logger.info("Estadísticas de comunicación reiniciadas")
