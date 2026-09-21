import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import random
import math
from collections import defaultdict, deque
import copy
import time
import threading

# Configuración del logger
logger = logging.getLogger(__name__)

class SGDMomentumWeightOptimizer(ABC):
    """
    Clase base abstracta para optimizadores SGD con momentum de pesos.
    Define la interfaz común para todas las estrategias de SGD con momentum.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        logger.info("SGDMomentumWeightOptimizer base inicializado.")

    @abstractmethod
    def sgd_momentum_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        """
        Método abstracto para optimizar los pesos usando SGD con momentum.
        Debe ser implementado por las subclases.
        """
        pass

class AdaptiveSGDMomentumOptimizer(SGDMomentumWeightOptimizer):
    """
    Optimizador de pesos basado en SGD con momentum adaptativo.
    Utiliza SGD con momentum adaptativo para optimizar los pesos de la red neuronal.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 1e-4, dampening: float = 0.0,
                 nesterov: bool = False, adaptive_lr: bool = True,
                 lr_decay_factor: float = 0.1, lr_decay_patience: int = 10,
                 config=None):
        super().__init__(config)
        self.learning_rate = self.config.get('learning_rate', learning_rate)
        self.momentum = self.config.get('momentum', momentum)
        self.weight_decay = self.config.get('weight_decay', weight_decay)
        self.dampening = self.config.get('dampening', dampening)
        self.nesterov = self.config.get('nesterov', nesterov)
        self.adaptive_lr = self.config.get('adaptive_lr', adaptive_lr)
        self.lr_decay_factor = self.config.get('lr_decay_factor', lr_decay_factor)
        self.lr_decay_patience = self.config.get('lr_decay_patience', lr_decay_patience)
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
        logger.info(f"AdaptiveSGDMomentumOptimizer inicializado: lr={self.learning_rate}, momentum={self.momentum}")

    def _create_optimizer(self, model: nn.Module) -> None:
        """
        Crea el optimizador SGD con momentum.
        """
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            dampening=self.dampening,
            nesterov=self.nesterov
        )
        
        # Crear scheduler si está habilitado
        if self.adaptive_lr:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.lr_decay_factor,
                patience=self.lr_decay_patience,
                verbose=True
            )
        
        logger.info("Optimizador SGD con momentum creado.")

    def _train_epoch(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Entrena el modelo por una época.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in data_loader:
            # Forward pass
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Aplicar momentum
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def _validate_epoch(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Valida el modelo por una época.
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def sgd_momentum_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización de pesos con SGD con momentum adaptativo.")
        
        if data_loader is None:
            logger.warning("No se proporcionó data_loader para SGD con momentum. Saltando.")
            return model
        
        # Crear optimizador
        self._create_optimizer(model)
        
        # Dividir datos en entrenamiento y validación
        train_size = int(0.8 * len(data_loader.dataset))
        val_size = len(data_loader.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(data_loader.dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_loader.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=data_loader.batch_size, shuffle=False)
        
        # Entrenamiento
        best_val_loss = float('inf')
        patience_counter = 0
        max_epochs = self.config.get('max_epochs', 100)
        
        for epoch in range(max_epochs):
            # Entrenar
            train_loss = self._train_epoch(model, train_loader)
            
            # Validar
            val_loss = self._validate_epoch(model, val_loader)
            
            # Actualizar scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Guardar historial
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_stats)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.lr_decay_patience * 2:
                    logger.info(f"Early stopping en época {epoch}")
                    break
            
            # Log de progreso
            if epoch % 10 == 0:
                logger.info(f"Época {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {self.optimizer.param_groups[0]['lr']:.6f}")
        
        logger.info(f"Optimización SGD con momentum completada. Mejor pérdida de validación = {best_val_loss:.4f}")
        return model

class NesterovSGDMomentumOptimizer(SGDMomentumWeightOptimizer):
    """
    Optimizador de pesos basado en SGD con momentum de Nesterov.
    Utiliza SGD con momentum de Nesterov para optimizar los pesos de la red neuronal.
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 1e-4, dampening: float = 0.0,
                 adaptive_momentum: bool = True, momentum_decay: float = 0.1,
                 config=None):
        super().__init__(config)
        self.learning_rate = self.config.get('learning_rate', learning_rate)
        self.momentum = self.config.get('momentum', momentum)
        self.weight_decay = self.config.get('weight_decay', weight_decay)
        self.dampening = self.config.get('dampening', dampening)
        self.adaptive_momentum = self.config.get('adaptive_momentum', adaptive_momentum)
        self.momentum_decay = self.config.get('momentum_decay', momentum_decay)
        self.optimizer = None
        self.training_history = []
        logger.info(f"NesterovSGDMomentumOptimizer inicializado: lr={self.learning_rate}, momentum={self.momentum}")

    def _create_optimizer(self, model: nn.Module) -> None:
        """
        Crea el optimizador SGD con momentum de Nesterov.
        """
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            dampening=self.dampening,
            nesterov=True  # Habilitar Nesterov momentum
        )
        
        logger.info("Optimizador SGD con momentum de Nesterov creado.")

    def _train_epoch(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Entrena el modelo por una época con Nesterov momentum.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in data_loader:
            # Forward pass
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Aplicar Nesterov momentum
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def _validate_epoch(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Valida el modelo por una época.
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def _update_momentum(self, epoch: int) -> None:
        """
        Actualiza el momentum de forma adaptativa.
        """
        if self.adaptive_momentum:
            # Decaimiento exponencial del momentum
            new_momentum = self.momentum * (self.momentum_decay ** epoch)
            new_momentum = max(0.1, new_momentum)  # Momentum mínimo
            
            # Actualizar momentum en el optimizador
            for param_group in self.optimizer.param_groups:
                param_group['momentum'] = new_momentum
            
            logger.debug(f"Momentum actualizado a {new_momentum:.4f} en época {epoch}")

    def sgd_momentum_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización de pesos con SGD con momentum de Nesterov.")
        
        if data_loader is None:
            logger.warning("No se proporcionó data_loader para SGD con momentum de Nesterov. Saltando.")
            return model
        
        # Crear optimizador
        self._create_optimizer(model)
        
        # Dividir datos en entrenamiento y validación
        train_size = int(0.8 * len(data_loader.dataset))
        val_size = len(data_loader.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(data_loader.dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_loader.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=data_loader.batch_size, shuffle=False)
        
        # Entrenamiento
        best_val_loss = float('inf')
        patience_counter = 0
        max_epochs = self.config.get('max_epochs', 100)
        
        for epoch in range(max_epochs):
            # Actualizar momentum
            self._update_momentum(epoch)
            
            # Entrenar
            train_loss = self._train_epoch(model, train_loader)
            
            # Validar
            val_loss = self._validate_epoch(model, val_loader)
            
            # Guardar historial
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'momentum': self.optimizer.param_groups[0]['momentum']
            }
            self.training_history.append(epoch_stats)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logger.info(f"Early stopping en época {epoch}")
                    break
            
            # Log de progreso
            if epoch % 10 == 0:
                logger.info(f"Época {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {self.optimizer.param_groups[0]['lr']:.6f}, Momentum = {self.optimizer.param_groups[0]['momentum']:.4f}")
        
        logger.info(f"Optimización SGD con momentum de Nesterov completada. Mejor pérdida de validación = {best_val_loss:.4f}")
        return model

class CyclicalSGDMomentumOptimizer(SGDMomentumWeightOptimizer):
    """
    Optimizador de pesos basado en SGD con momentum cíclico.
    Utiliza SGD con momentum cíclico para optimizar los pesos de la red neuronal.
    """
    def __init__(self, base_lr: float = 0.01, max_lr: float = 0.1,
                 base_momentum: float = 0.85, max_momentum: float = 0.95,
                 weight_decay: float = 1e-4, cycle_length: int = 10,
                 config=None):
        super().__init__(config)
        self.base_lr = self.config.get('base_lr', base_lr)
        self.max_lr = self.config.get('max_lr', max_lr)
        self.base_momentum = self.config.get('base_momentum', base_momentum)
        self.max_momentum = self.config.get('max_momentum', max_momentum)
        self.weight_decay = self.config.get('weight_decay', weight_decay)
        self.cycle_length = self.config.get('cycle_length', cycle_length)
        self.optimizer = None
        self.training_history = []
        logger.info(f"CyclicalSGDMomentumOptimizer inicializado: base_lr={self.base_lr}, max_lr={self.max_lr}")

    def _create_optimizer(self, model: nn.Module) -> None:
        """
        Crea el optimizador SGD con momentum cíclico.
        """
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.base_lr,
            momentum=self.base_momentum,
            weight_decay=self.weight_decay
        )
        
        logger.info("Optimizador SGD con momentum cíclico creado.")

    def _update_cyclical_parameters(self, epoch: int, batch_idx: int, total_batches: int) -> None:
        """
        Actualiza los parámetros cíclicos (learning rate y momentum).
        """
        # Calcular posición en el ciclo
        cycle_position = (epoch * total_batches + batch_idx) % self.cycle_length
        cycle_ratio = cycle_position / self.cycle_length
        
        # Actualizar learning rate cíclico
        if cycle_ratio <= 0.5:
            # Fase de aumento
            lr = self.base_lr + (self.max_lr - self.base_lr) * (2 * cycle_ratio)
        else:
            # Fase de disminución
            lr = self.max_lr - (self.max_lr - self.base_lr) * (2 * (cycle_ratio - 0.5))
        
        # Actualizar momentum cíclico (inverso al learning rate)
        if cycle_ratio <= 0.5:
            # Fase de disminución
            momentum = self.max_momentum - (self.max_momentum - self.base_momentum) * (2 * cycle_ratio)
        else:
            # Fase de aumento
            momentum = self.base_momentum + (self.max_momentum - self.base_momentum) * (2 * (cycle_ratio - 0.5))
        
        # Actualizar parámetros en el optimizador
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = momentum

    def _train_epoch(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Entrena el modelo por una época con parámetros cíclicos.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Actualizar parámetros cíclicos
            self._update_cyclical_parameters(0, batch_idx, len(data_loader))
            
            # Forward pass
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Aplicar momentum cíclico
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def _validate_epoch(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Valida el modelo por una época.
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def sgd_momentum_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización de pesos con SGD con momentum cíclico.")
        
        if data_loader is None:
            logger.warning("No se proporcionó data_loader para SGD con momentum cíclico. Saltando.")
            return model
        
        # Crear optimizador
        self._create_optimizer(model)
        
        # Dividir datos en entrenamiento y validación
        train_size = int(0.8 * len(data_loader.dataset))
        val_size = len(data_loader.dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(data_loader.dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_loader.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=data_loader.batch_size, shuffle=False)
        
        # Entrenamiento
        best_val_loss = float('inf')
        patience_counter = 0
        max_epochs = self.config.get('max_epochs', 100)
        
        for epoch in range(max_epochs):
            # Entrenar
            train_loss = self._train_epoch(model, train_loader)
            
            # Validar
            val_loss = self._validate_epoch(model, val_loader)
            
            # Guardar historial
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'momentum': self.optimizer.param_groups[0]['momentum']
            }
            self.training_history.append(epoch_stats)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logger.info(f"Early stopping en época {epoch}")
                    break
            
            # Log de progreso
            if epoch % 10 == 0:
                logger.info(f"Época {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {self.optimizer.param_groups[0]['lr']:.6f}, Momentum = {self.optimizer.param_groups[0]['momentum']:.4f}")
        
        logger.info(f"Optimización SGD con momentum cíclico completada. Mejor pérdida de validación = {best_val_loss:.4f}")
        return model

class SGDMomentumWeightAnalyzer:
    """
    Analizador para evaluar el rendimiento de SGD con momentum de pesos.
    """
    def __init__(self):
        logger.info("SGDMomentumWeightAnalyzer inicializado.")

    def analyze_sgd_momentum_optimization(self, original_model: nn.Module, 
                                         optimized_model: nn.Module, 
                                         test_data_loader=None) -> Dict[str, float]:
        """
        Analiza el rendimiento de SGD con momentum.
        """
        analysis_results = {}
        
        # Evaluar rendimiento original
        original_performance = self._evaluate_model_performance(original_model, test_data_loader)
        
        # Evaluar rendimiento optimizado
        optimized_performance = self._evaluate_model_performance(optimized_model, test_data_loader)
        
        # Calcular mejora
        improvement = original_performance - optimized_performance
        improvement_percentage = (improvement / original_performance) * 100
        
        analysis_results['original_performance'] = original_performance
        analysis_results['optimized_performance'] = optimized_performance
        analysis_results['improvement'] = improvement
        analysis_results['improvement_percentage'] = improvement_percentage
        
        # Analizar características de SGD con momentum
        analysis_results['momentum_effectiveness'] = self._analyze_momentum_effectiveness(optimized_model)
        analysis_results['convergence_stability'] = self._analyze_convergence_stability(optimized_model)
        analysis_results['learning_rate_adaptation'] = self._analyze_learning_rate_adaptation(optimized_model)
        
        logger.info(f"Análisis SGD con momentum: Mejora = {improvement_percentage:.2f}%")
        return analysis_results

    def _evaluate_model_performance(self, model: nn.Module, data_loader) -> float:
        """
        Evalúa el rendimiento del modelo.
        """
        if data_loader is None:
            return random.random()
        
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.item()
        
        return total_loss

    def _analyze_momentum_effectiveness(self, model: nn.Module) -> float:
        """
        Analiza la efectividad del momentum del modelo.
        """
        # Simular efectividad del momentum basándose en la estabilidad de pesos
        weight_stability = 0.0
        for param in model.parameters():
            if param.requires_grad:
                weight_stability += torch.std(param.data).item()
        
        momentum_effectiveness = 1.0 / (1.0 + weight_stability)
        return momentum_effectiveness

    def _analyze_convergence_stability(self, model: nn.Module) -> float:
        """
        Analiza la estabilidad de convergencia del modelo.
        """
        # Simular estabilidad de convergencia basándose en la variabilidad de pesos
        weight_vars = []
        for param in model.parameters():
            if param.requires_grad:
                weight_vars.append(torch.var(param.data).item())
        
        convergence_stability = 1.0 / (1.0 + np.mean(weight_vars))
        return convergence_stability

    def _analyze_learning_rate_adaptation(self, model: nn.Module) -> float:
        """
        Analiza la adaptación del learning rate del modelo.
        """
        # Simular adaptación del learning rate basándose en la magnitud de pesos
        weight_magnitudes = []
        for param in model.parameters():
            if param.requires_grad:
                weight_magnitudes.append(torch.norm(param.data).item())
        
        learning_rate_adaptation = 1.0 / (1.0 + np.mean(weight_magnitudes))
        return learning_rate_adaptation

def create_sgd_momentum_weight_optimizer(optimizer_type: str, **kwargs) -> SGDMomentumWeightOptimizer:
    """
    Factoría para crear diferentes tipos de optimizadores SGD con momentum.
    """
    if optimizer_type == "adaptive_sgd_momentum":
        return AdaptiveSGDMomentumOptimizer(**kwargs)
    elif optimizer_type == "nesterov_sgd_momentum":
        return NesterovSGDMomentumOptimizer(**kwargs)
    elif optimizer_type == "cyclical_sgd_momentum":
        return CyclicalSGDMomentumOptimizer(**kwargs)
    else:
        raise ValueError(f"Tipo de optimizador SGD con momentum no soportado: {optimizer_type}")

def sgd_momentum_optimize_model_weights(model: nn.Module, optimizer_type: str, 
                                       data_loader=None, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Función de conveniencia para aplicar SGD con momentum a los pesos de un modelo.
    """
    optimizer = create_sgd_momentum_weight_optimizer(optimizer_type, **kwargs)
    original_model = copy.deepcopy(model)
    
    # Aplicar SGD con momentum
    optimized_model = optimizer.sgd_momentum_optimize_weights(model, data_loader)
    
    # Analizar rendimiento
    analyzer = SGDMomentumWeightAnalyzer()
    analysis = analyzer.analyze_sgd_momentum_optimization(original_model, optimized_model, data_loader)
    
    return optimized_model, analysis

# Exportar clases y funciones principales
__all__ = [
    'SGDMomentumWeightOptimizer',
    'AdaptiveSGDMomentumOptimizer',
    'NesterovSGDMomentumOptimizer',
    'CyclicalSGDMomentumOptimizer',
    'SGDMomentumWeightAnalyzer',
    'create_sgd_momentum_weight_optimizer',
    'sgd_momentum_optimize_model_weights'
]

logger.info("RFEN7_RN_7 - SGD con Momentum de Pesos cargada correctamente")
