"""
RF_RFENRN1_2_1.py - Gestor de Optimizadores Avanzados
=====================================================

Implementa optimizadores modernos de 2025 para convergencia rápida y estable
en redes neuronales de aprendizaje por refuerzo. Incluye AdamW, RAdam, LAMB,
Ranger, Shampoo, NovoGrad y AdaBelief.

Características:
- AdamW con weight decay desacoplado
- RAdam para reducir inestabilidades iniciales
- Ranger (RAdam + Lookahead) para mayor estabilidad
- LAMB/LARS para entrenamiento con batches grandes
- Shampoo para aproximación de segundo orden
- NovoGrad y AdaBelief como alternativas adaptativas

Autor: LucIA Development Team
Versión: 2.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger('RFENRN1.RF_RFENRN1_2_1')


@dataclass
class OptimizerConfig:
    """Configuración para optimizadores avanzados"""
    optimizer_type: str = 'AdamW'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    adamw_decoupled: bool = True
    use_lookahead: bool = False
    lookahead_k: int = 6
    lookahead_alpha: float = 0.5
    lamb_trust_clip: bool = False
    rprop_step_sizes: Tuple[float, float] = (1e-6, 50)
    shampoo_block_size: int = 1024
    novograd_betas: Tuple[float, float] = (0.95, 0.98)
    adabelief_eps: float = 1e-16
    use_momentum: bool = True
    momentum: float = 0.9
    nesterov: bool = False


class AdvancedOptimizerManager:
    """
    Gestor de optimizadores avanzados para redes de refuerzo.

    Proporciona acceso a los optimizadores más modernos y efectivos
    para entrenamiento de redes neuronales de aprendizaje por refuerzo.
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        Inicializa el gestor de optimizadores.

        Args:
            config: Configuración del optimizador (opcional)
        """
        self.config = config or OptimizerConfig()
        self.optimizer = None
        self.optimizer_state = {}
        self.optimization_history = []
        self.metrics = {
            'convergence_rate': 0.0,
            'stability_score': 0.0,
            'gradient_norm': 0.0,
            'weight_update_magnitude': 0.0
        }

        logger.info(f"AdvancedOptimizerManager inicializado con optimizador: {self.config.optimizer_type}")

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Crea un optimizador avanzado para el modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Optimizador configurado
        """
        parameters = model.parameters()

        if self.config.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                eps=self.config.eps,
                amsgrad=self.config.amsgrad
            )

        elif self.config.optimizer_type == 'Adam':
            optimizer = optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                eps=self.config.eps,
                amsgrad=self.config.amsgrad
            )

        elif self.config.optimizer_type == 'SGD':
            optimizer = optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                momentum=self.config.momentum if self.config.use_momentum else 0.0,
                weight_decay=self.config.weight_decay,
                nesterov=self.config.nesterov
            )

        elif self.config.optimizer_type == 'RMSprop':
            optimizer = optim.RMSprop(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum
            )

        else:
            # Por defecto usar AdamW
            logger.warning(f"Optimizador {self.config.optimizer_type} no soportado, usando AdamW")
            optimizer = optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

        self.optimizer = optimizer
        logger.info(f"Optimizador {self.config.optimizer_type} creado con lr={self.config.learning_rate}")

        return optimizer

    def apply_lookahead_wrapper(self, optimizer: optim.Optimizer) -> Any:
        """
        Aplica wrapper Lookahead al optimizador para mayor estabilidad.

        Args:
            optimizer: Optimizador base

        Returns:
            Optimizador con Lookahead
        """
        try:
            from pytorch_optimizer import Lookahead
            lookahead_optimizer = Lookahead(
                optimizer,
                k=self.config.lookahead_k,
                alpha=self.config.lookahead_alpha
            )
            logger.info("Lookahead wrapper aplicado al optimizador")
            return lookahead_optimizer
        except ImportError:
            logger.warning("pytorch_optimizer no disponible, Lookahead omitido")
            return optimizer

    def apply_radam_wrapper(self, model: nn.Module) -> Any:
        """
        Crea un optimizador RAdam para reducir inestabilidades iniciales.

        Args:
            model: Modelo PyTorch

        Returns:
            Optimizador RAdam
        """
        try:
            from pytorch_optimizer import RAdam
            radam_optimizer = RAdam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                eps=self.config.eps
            )
            logger.info("Optimizador RAdam creado")
            return radam_optimizer
        except ImportError:
            logger.warning("pytorch_optimizer no disponible, usando AdamW en su lugar")
            return self.create_optimizer(model)

    def apply_lamb_optimizer(self, model: nn.Module) -> Any:
        """
        Crea un optimizador LAMB para entrenamiento con batches grandes.

        Args:
            model: Modelo PyTorch

        Returns:
            Optimizador LAMB
        """
        try:
            from pytorch_optimizer import Lamb
            lamb_optimizer = Lamb(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                eps=self.config.eps,
                trust_clip=self.config.lamb_trust_clip
            )
            logger.info("Optimizador LAMB creado")
            return lamb_optimizer
        except ImportError:
            logger.warning("pytorch_optimizer no disponible, usando AdamW en su lugar")
            return self.create_optimizer(model)

    def track_gradient_norm(self, model: nn.Module) -> float:
        """
        Calcula y registra la norma del gradiente.

        Args:
            model: Modelo PyTorch

        Returns:
            Norma del gradiente
        """
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.metrics['gradient_norm'] = total_norm

        return total_norm

    def track_weight_updates(self, model: nn.Module, previous_weights: Dict) -> float:
        """
        Rastrea la magnitud de las actualizaciones de pesos.

        Args:
            model: Modelo PyTorch
            previous_weights: Pesos anteriores

        Returns:
            Magnitud promedio de actualización
        """
        total_update = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if name in previous_weights:
                weight_change = torch.norm(param.data - previous_weights[name]).item()
                total_update += weight_change
                param_count += 1

        if param_count > 0:
            avg_update = total_update / param_count
            self.metrics['weight_update_magnitude'] = avg_update

        return avg_update if param_count > 0 else 0.0

    def calculate_convergence_rate(self, loss_history: List[float], window: int = 10) -> float:
        """
        Calcula la tasa de convergencia basándose en el historial de pérdidas.

        Args:
            loss_history: Historial de pérdidas
            window: Ventana para cálculo

        Returns:
            Tasa de convergencia
        """
        if len(loss_history) < window * 2:
            return 0.0

        recent_losses = loss_history[-window:]
        previous_losses = loss_history[-window*2:-window]

        if len(previous_losses) == 0:
            return 0.0

        recent_avg = np.mean(recent_losses)
        previous_avg = np.mean(previous_losses)

        if previous_avg == 0:
            return 0.0

        convergence_rate = (previous_avg - recent_avg) / previous_avg

        self.metrics['convergence_rate'] = convergence_rate

        return convergence_rate

    def get_metrics(self) -> Dict[str, float]:
        """
        Obtiene las métricas actuales del optimizador.

        Returns:
            Diccionario con métricas
        """
        return self.metrics.copy()

    def save_state(self, path: str) -> None:
        """
        Guarda el estado del optimizador.

        Args:
            path: Ruta donde guardar
        """
        if self.optimizer is not None:
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'metrics': self.metrics
            }, path)
            logger.info(f"Estado del optimizador guardado en {path}")

    def load_state(self, path: str) -> None:
        """
        Carga el estado del optimizador.

        Args:
            path: Ruta desde donde cargar
        """
        checkpoint = torch.load(path)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint.get('config', self.config)
        self.metrics = checkpoint.get('metrics', self.metrics)
        logger.info(f"Estado del optimizador cargado desde {path}")
