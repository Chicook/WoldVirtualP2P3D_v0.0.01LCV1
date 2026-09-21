"""
RFEN8_RN_5.py - AdaBound Optimizer Adaptativo
==============================================

Implementación del optimizador AdaBound que combina la adaptabilidad de Adam
con la estabilidad de SGD mediante límites adaptativos en el learning rate.

Características principales:
- Transición automática de Adam a SGD
- Límites adaptativos en learning rate
- Mejor convergencia y estabilidad
- Manejo inteligente de hiperparámetros
- Robustez mejorada

Referencias:
- Luo, L., et al. "Adaptive Gradient Methods with Dynamic Bound of Learning Rate"
- Implementación basada en el paper original
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time
import copy
from ..RFENRN8 import BaseAdvancedOptimizer, AdvancedOptimizerConfig, OptimizationResult, OptimizationMetrics

logger = logging.getLogger(__name__)

class AdaBoundWeightOptimizer(BaseAdvancedOptimizer):
    """Optimizador AdaBound con límites adaptativos"""
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.adabound_final_lr = config.adabound_final_lr
        self.adabound_gamma = config.adabound_gamma
        self.bound_history = []
        self.transition_tracker = []
        self.adaptive_bounds = True
        self.bound_decay_factor = 0.99
        
        logger.info(f"AdaBoundWeightOptimizer inicializado con final_lr={self.adabound_final_lr}, gamma={self.adabound_gamma}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador AdaBound"""
        try:
            adabound_optimizer = AdaBound(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                final_lr=self.adabound_final_lr,
                gamma=self.adabound_gamma,
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay
            )
            
            self.optimizer = adabound_optimizer
            logger.info("Optimizador AdaBound creado exitosamente")
            return adabound_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador AdaBound: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos del modelo usando AdaBound"""
        try:
            logger.info("Iniciando optimización AdaBound")
            start_time = time.time()
            
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            optimizer = self.create_optimizer(model)
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            
            model.train()
            loss_history = []
            accuracy_history = []
            bound_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    loss = self._adabound_step(model, data, target, optimizer, criterion)
                    epoch_losses.append(loss.item())
                    
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    if hasattr(optimizer, 'current_bound'):
                        bound_history.append(optimizer.current_bound)
                
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
            
            # Análisis de límites adaptativos
            bound_analysis = self._analyze_adaptive_bounds(bound_history)
            
            metrics = OptimizationMetrics(
                optimizer_name="AdaBound",
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
                optimization_robustness=bound_analysis['robustness'],
                overall_score=self._calculate_adabound_score(initial_metrics, final_metrics, bound_analysis),
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
                performance_analysis={'bound_analysis': bound_analysis},
                recommendations=self._generate_adabound_recommendations(metrics, bound_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización AdaBound completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización AdaBound: {e}")
            return OptimizationResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights=None,
                convergence_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(e)
            )
    
    def _adabound_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                       optimizer: 'AdaBound', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización AdaBound"""
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss
        except Exception as e:
            logger.error(f"Error en paso AdaBound: {e}")
            raise
    
    def _analyze_adaptive_bounds(self, bound_history: List[float]) -> Dict:
        """Analiza los límites adaptativos"""
        try:
            if not bound_history:
                return {'robustness': 0.0, 'bound_effectiveness': 0.0, 'adaptation_smoothness': 0.0}
            
            # Calcular efectividad de los límites
            mean_bound = np.mean(bound_history)
            std_bound = np.std(bound_history)
            bound_effectiveness = max(0.0, 1.0 - std_bound / max(mean_bound, 1e-8))
            
            # Calcular suavidad de adaptación
            if len(bound_history) > 1:
                adaptations = np.diff(bound_history)
                adaptation_smoothness = max(0.0, 1.0 - np.std(adaptations) / max(np.mean(np.abs(adaptations)), 1e-8))
            else:
                adaptation_smoothness = 0.0
            
            # Calcular robustez general
            robustness = (bound_effectiveness + adaptation_smoothness) / 2.0
            
            return {
                'robustness': robustness,
                'bound_effectiveness': bound_effectiveness,
                'adaptation_smoothness': adaptation_smoothness,
                'mean_bound': mean_bound,
                'bound_variance': std_bound
            }
            
        except Exception as e:
            logger.error(f"Error analizando límites adaptativos: {e}")
            return {'robustness': 0.0, 'bound_effectiveness': 0.0, 'adaptation_smoothness': 0.0}
    
    def _calculate_adabound_score(self, initial_metrics: Dict, final_metrics: Dict, 
                                 bound_analysis: Dict) -> float:
        """Calcula el score específico de AdaBound"""
        try:
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            robustness = bound_analysis.get('robustness', 0.0)
            bound_effectiveness = bound_analysis.get('bound_effectiveness', 0.0)
            
            adabound_score = (
                loss_improvement * 0.3 +
                accuracy_improvement * 0.3 +
                robustness * 0.2 +
                bound_effectiveness * 0.2
            )
            
            return max(0.0, min(1.0, adabound_score))
            
        except Exception as e:
            logger.error(f"Error calculando score AdaBound: {e}")
            return 0.0
    
    def _generate_adabound_recommendations(self, metrics: OptimizationMetrics, 
                                          bound_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para AdaBound"""
        recommendations = []
        
        try:
            if bound_analysis.get('robustness', 0.0) < 0.7:
                recommendations.append("La robustez de límites es baja, considerar ajustar gamma y final_lr")
            
            if bound_analysis.get('adaptation_smoothness', 0.0) < 0.6:
                recommendations.append("La adaptación de límites no es suave, considerar ajustar gamma")
            
            if metrics.convergence_speed < 0.01:
                recommendations.append("La convergencia es lenta, considerar aumentar la tasa de aprendizaje inicial")
            
            if metrics.training_stability < 0.7:
                recommendations.append("La estabilidad del entrenamiento es baja, considerar reducir el learning rate")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones AdaBound: {e}")
        
        return recommendations

class AdaBound(torch.optim.Optimizer):
    """Implementación del optimizador AdaBound"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3, eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps, weight_decay=weight_decay)
        super(AdaBound, self).__init__(params, defaults)
        
        self.current_bound = 0.0
    
    def step(self, closure=None):
        """Paso de optimización AdaBound"""
        loss = None
        if closure is not None:
            loss = closure()
        
        bounds = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBound no soporta gradientes dispersos')
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Actualizar estimaciones
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Calcular bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Calcular límites adaptativos
                step_size = group['lr'] / bias_correction1
                final_lr = group['final_lr'] * group['lr'] / group['lr']
                
                # Calcular límites
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                
                # Aplicar límites al step size
                step_size = torch.clamp(step_size, lower_bound, upper_bound)
                
                # Calcular actualización
                denom = (exp_avg_sq.div(bias_correction2).sqrt().add_(group['eps']))
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                bounds.append(step_size.item())
        
        if bounds:
            self.current_bound = np.mean(bounds)
        
        return loss

def create_adabound_optimizer(config: AdvancedOptimizerConfig = None) -> AdaBoundWeightOptimizer:
    """Crea un optimizador AdaBound"""
    return AdaBoundWeightOptimizer(config or AdvancedOptimizerConfig())

def analyze_adabound_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de AdaBound en un modelo"""
    try:
        optimizer = AdaBoundWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'bound_analysis': result.performance_analysis.get('bound_analysis', {})
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento AdaBound: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_5.py - AdaBound Optimizer cargado exitosamente")
