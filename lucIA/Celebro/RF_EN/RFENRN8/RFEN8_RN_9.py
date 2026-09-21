"""
RFEN8_RN_9.py - QHAdam (Quasi-Hyperbolic Adam) Optimizer
========================================================

Implementación del optimizador QHAdam que combina momentum cuasi-hiperbólico
con Adam para mejorar la convergencia y estabilidad.

Características principales:
- Momentum cuasi-hiperbólico
- Combinación de SGD y Adam
- Mejor convergencia en problemas complejos
- Parámetros adaptativos nu1 y nu2
- Robustez mejorada

Referencias:
- Ma, J., & Yarats, D. "Quasi-Hyperbolic Momentum and Adam for Deep Learning"
- Implementación basada en el paper original
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time
import copy
from ..RFENRN8 import BaseAdvancedOptimizer, AdvancedOptimizerConfig, OptimizationResult, OptimizationMetrics

logger = logging.getLogger(__name__)

class QHAdamWeightOptimizer(BaseAdvancedOptimizer):
    """Optimizador QHAdam con momentum cuasi-hiperbólico"""
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.qhadam_nu1 = config.qhadam_nu1
        self.qhadam_nu2 = config.qhadam_nu2
        self.momentum_history = []
        self.hyperbolic_metrics = {}
        self.adaptive_nu = True
        self.nu_decay_factor = 0.99
        
        logger.info(f"QHAdamWeightOptimizer inicializado con nu1={self.qhadam_nu1}, nu2={self.qhadam_nu2}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador QHAdam"""
        try:
            qhadam_optimizer = QHAdam(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                nu1=self.qhadam_nu1,
                nu2=self.qhadam_nu2,
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay
            )
            
            self.optimizer = qhadam_optimizer
            logger.info("Optimizador QHAdam creado exitosamente")
            return qhadam_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador QHAdam: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos del modelo usando QHAdam"""
        try:
            logger.info("Iniciando optimización QHAdam")
            start_time = time.time()
            
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            optimizer = self.create_optimizer(model)
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            
            model.train()
            loss_history = []
            accuracy_history = []
            momentum_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    loss = self._qhadam_step(model, data, target, optimizer, criterion)
                    epoch_losses.append(loss.item())
                    
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    if hasattr(optimizer, 'momentum_magnitude'):
                        momentum_history.append(optimizer.momentum_magnitude)
                
                avg_loss = np.mean(epoch_losses)
                avg_accuracy = np.mean(epoch_accuracies)
                loss_history.append(avg_loss)
                accuracy_history.append(avg_accuracy)
                
                if self._check_convergence(loss_history):
                    logger.info(f"Convergencia alcanzada en época {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Época {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
            
            final_metrics = self._evaluate_model(model, data_loader, criterion)
            optimization_time = time.time() - start_time
            
            # Análisis de momentum cuasi-hiperbólico
            hyperbolic_analysis = self._analyze_hyperbolic_momentum(momentum_history)
            
            metrics = OptimizationMetrics(
                optimizer_name="QHAdam",
                initial_loss=initial_metrics['loss'],
                final_loss=final_metrics['loss'],
                convergence_iterations=len(loss_history),
                convergence_speed=1.0 / max(len(loss_history), 1),
                generalization_improvement=final_metrics['accuracy'] - initial_metrics['accuracy'],
                computational_efficiency=1.0 / optimization_time,
                memory_usage=self._estimate_memory_usage(model),
                training_stability=1.0 - np.std(loss_history[-10:]) / max(np.mean(loss_history[-10:]), 1e-8),
                test_accuracy=final_metrics['accuracy'],
                loss_reduction=(initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8),
                gradient_norm=self._calculate_gradient_norm(model),
                weight_magnitude=self._calculate_weight_magnitude(model),
                optimization_robustness=hyperbolic_analysis['robustness'],
                overall_score=self._calculate_qhadam_score(initial_metrics, final_metrics, hyperbolic_analysis),
                optimization_time=optimization_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            result = OptimizationResult(
                success=True,
                optimized_model=model,
                metrics=metrics,
                optimization_history=loss_history,
                best_weights=copy.deepcopy(model.state_dict()),
                convergence_analysis=self.analyze_convergence(loss_history),
                performance_analysis={'hyperbolic_analysis': hyperbolic_analysis},
                recommendations=self._generate_qhadam_recommendations(metrics, hyperbolic_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización QHAdam completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización QHAdam: {e}")
            return OptimizationResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights=None,
                convergence_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(e)
            )
    
    def _qhadam_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                     optimizer: 'QHAdam', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización QHAdam"""
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss
        except Exception as e:
            logger.error(f"Error en paso QHAdam: {e}")
            raise
    
    def _analyze_hyperbolic_momentum(self, momentum_history: List[float]) -> Dict:
        """Analiza el momentum cuasi-hiperbólico"""
        try:
            if not momentum_history:
                return {'robustness': 0.0, 'momentum_effectiveness': 0.0, 'hyperbolic_stability': 0.0}
            
            # Calcular efectividad del momentum
            mean_momentum = np.mean(momentum_history)
            std_momentum = np.std(momentum_history)
            momentum_effectiveness = max(0.0, 1.0 - std_momentum / max(mean_momentum, 1e-8))
            
            # Calcular estabilidad hiperbólica
            hyperbolic_stability = max(0.0, 1.0 - std_momentum / max(mean_momentum, 1e-8))
            
            # Calcular robustez
            robustness = (momentum_effectiveness + hyperbolic_stability) / 2.0
            
            return {
                'robustness': robustness,
                'momentum_effectiveness': momentum_effectiveness,
                'hyperbolic_stability': hyperbolic_stability,
                'mean_momentum': mean_momentum,
                'momentum_variance': std_momentum
            }
            
        except Exception as e:
            logger.error(f"Error analizando momentum cuasi-hiperbólico: {e}")
            return {'robustness': 0.0, 'momentum_effectiveness': 0.0, 'hyperbolic_stability': 0.0}
    
    def _calculate_qhadam_score(self, initial_metrics: Dict, final_metrics: Dict, 
                                hyperbolic_analysis: Dict) -> float:
        """Calcula el score específico de QHAdam"""
        try:
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            robustness = hyperbolic_analysis.get('robustness', 0.0)
            momentum_effectiveness = hyperbolic_analysis.get('momentum_effectiveness', 0.0)
            
            qhadam_score = (
                loss_improvement * 0.3 +
                accuracy_improvement * 0.3 +
                robustness * 0.2 +
                momentum_effectiveness * 0.2
            )
            
            return max(0.0, min(1.0, qhadam_score))
            
        except Exception as e:
            logger.error(f"Error calculando score QHAdam: {e}")
            return 0.0
    
    def _generate_qhadam_recommendations(self, metrics: OptimizationMetrics, 
                                         hyperbolic_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para QHAdam"""
        recommendations = []
        
        try:
            if hyperbolic_analysis.get('momentum_effectiveness', 0.0) < 0.7:
                recommendations.append("La efectividad del momentum es baja, considerar ajustar nu1 y nu2")
            
            if hyperbolic_analysis.get('hyperbolic_stability', 0.0) < 0.6:
                recommendations.append("La estabilidad hiperbólica es baja, considerar ajustar los parámetros")
            
            if metrics.convergence_speed < 0.01:
                recommendations.append("La convergencia es lenta, considerar aumentar la tasa de aprendizaje")
            
            if metrics.training_stability < 0.7:
                recommendations.append("La estabilidad del entrenamiento es baja, considerar reducir el learning rate")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones QHAdam: {e}")
        
        return recommendations

class QHAdam(torch.optim.Optimizer):
    """Implementación del optimizador QHAdam"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), nu1=0.7, nu2=1.0, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, nu1=nu1, nu2=nu2, eps=eps, weight_decay=weight_decay)
        super(QHAdam, self).__init__(params, defaults)
        
        self.momentum_magnitude = 0.0
    
    def step(self, closure=None):
        """Paso de optimización QHAdam"""
        loss = None
        if closure is not None:
            loss = closure()
        
        momentum_magnitudes = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('QHAdam no soporta gradientes dispersos')
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                nu1, nu2 = group['nu1'], group['nu2']
                state['step'] += 1
                
                # Actualizar estimaciones
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Calcular bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # QHAdam: momentum cuasi-hiperbólico
                denom = (exp_avg_sq.div(bias_correction2).sqrt().add_(group['eps']))
                step_size = group['lr'] / bias_correction1
                
                # Aplicar momentum cuasi-hiperbólico
                momentum_term = nu1 * exp_avg + (1 - nu1) * grad
                p.data.addcdiv_(momentum_term, denom, value=-step_size * nu2)
                
                # Aplicar weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                momentum_magnitudes.append(torch.norm(momentum_term).item())
        
        if momentum_magnitudes:
            self.momentum_magnitude = np.mean(momentum_magnitudes)
        
        return loss

def create_qhadam_optimizer(config: AdvancedOptimizerConfig = None) -> QHAdamWeightOptimizer:
    """Crea un optimizador QHAdam"""
    return QHAdamWeightOptimizer(config or AdvancedOptimizerConfig())

def analyze_qhadam_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                              criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de QHAdam en un modelo"""
    try:
        optimizer = QHAdamWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'hyperbolic_analysis': result.performance_analysis.get('hyperbolic_analysis', {})
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento QHAdam: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_9.py - QHAdam Optimizer cargado exitosamente")
