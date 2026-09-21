"""
RFEN8_RN_4.py - RAdam (Rectified Adam) Optimizer
================================================

Implementación del optimizador RAdam que corrige el bias en Adam durante
las primeras iteraciones, mejorando la estabilidad y convergencia.

Características principales:
- Corrección de bias adaptativa para Adam
- Transición suave de SGD a Adam
- Mejor estabilidad en las primeras iteraciones
- Convergencia más rápida y estable
- Manejo inteligente de la varianza

Referencias:
- Liu, L., et al. "On the Variance of the Adaptive Learning Rate and Beyond"
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

class RAdamWeightOptimizer(BaseAdvancedOptimizer):
    """Optimizador RAdam con corrección de bias adaptativa"""
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.radam_beta1 = config.radam_beta1
        self.radam_beta2 = config.radam_beta2
        self.radam_eps = config.radam_eps
        self.bias_correction_history = []
        self.variance_tracker = []
        self.transition_metrics = {}
        self.adaptive_warmup = True
        self.warmup_steps = config.warmup_steps
        
        logger.info(f"RAdamWeightOptimizer inicializado con beta1={self.radam_beta1}, beta2={self.radam_beta2}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador RAdam"""
        try:
            radam_optimizer = RAdam(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.radam_beta1, self.radam_beta2),
                eps=self.radam_eps,
                weight_decay=self.config.weight_decay
            )
            
            self.optimizer = radam_optimizer
            logger.info("Optimizador RAdam creado exitosamente")
            return radam_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador RAdam: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos del modelo usando RAdam"""
        try:
            logger.info("Iniciando optimización RAdam")
            start_time = time.time()
            
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            optimizer = self.create_optimizer(model)
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            
            model.train()
            loss_history = []
            accuracy_history = []
            bias_correction_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    loss = self._radam_step(model, data, target, optimizer, criterion)
                    epoch_losses.append(loss.item())
                    
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    if hasattr(optimizer, 'bias_correction_factor'):
                        bias_correction_history.append(optimizer.bias_correction_factor)
                
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
            
            # Análisis de transición SGD-Adam
            transition_analysis = self._analyze_transition(bias_correction_history)
            
            metrics = OptimizationMetrics(
                optimizer_name="RAdam",
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
                optimization_robustness=transition_analysis['stability'],
                overall_score=self._calculate_radam_score(initial_metrics, final_metrics, transition_analysis),
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
                performance_analysis={'transition_analysis': transition_analysis},
                recommendations=self._generate_radam_recommendations(metrics, transition_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización RAdam completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización RAdam: {e}")
            return OptimizationResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights=None,
                convergence_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(e)
            )
    
    def _radam_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                    optimizer: 'RAdam', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización RAdam"""
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss
        except Exception as e:
            logger.error(f"Error en paso RAdam: {e}")
            raise
    
    def _analyze_transition(self, bias_correction_history: List[float]) -> Dict:
        """Analiza la transición de SGD a Adam"""
        try:
            if not bias_correction_history:
                return {'stability': 0.0, 'transition_smoothness': 0.0, 'bias_correction_effectiveness': 0.0}
            
            # Calcular estabilidad de la transición
            mean_correction = np.mean(bias_correction_history)
            std_correction = np.std(bias_correction_history)
            stability = max(0.0, 1.0 - std_correction / max(mean_correction, 1e-8))
            
            # Calcular suavidad de la transición
            if len(bias_correction_history) > 1:
                transitions = np.diff(bias_correction_history)
                transition_smoothness = max(0.0, 1.0 - np.std(transitions) / max(np.mean(np.abs(transitions)), 1e-8))
            else:
                transition_smoothness = 0.0
            
            # Efectividad de la corrección de bias
            bias_correction_effectiveness = min(1.0, mean_correction)
            
            return {
                'stability': stability,
                'transition_smoothness': transition_smoothness,
                'bias_correction_effectiveness': bias_correction_effectiveness,
                'mean_correction': mean_correction,
                'correction_variance': std_correction
            }
            
        except Exception as e:
            logger.error(f"Error analizando transición: {e}")
            return {'stability': 0.0, 'transition_smoothness': 0.0, 'bias_correction_effectiveness': 0.0}
    
    def _calculate_radam_score(self, initial_metrics: Dict, final_metrics: Dict, 
                              transition_analysis: Dict) -> float:
        """Calcula el score específico de RAdam"""
        try:
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            stability = transition_analysis.get('stability', 0.0)
            transition_smoothness = transition_analysis.get('transition_smoothness', 0.0)
            
            radam_score = (
                loss_improvement * 0.3 +
                accuracy_improvement * 0.3 +
                stability * 0.2 +
                transition_smoothness * 0.2
            )
            
            return max(0.0, min(1.0, radam_score))
            
        except Exception as e:
            logger.error(f"Error calculando score RAdam: {e}")
            return 0.0
    
    def _generate_radam_recommendations(self, metrics: OptimizationMetrics, 
                                       transition_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para RAdam"""
        recommendations = []
        
        try:
            if transition_analysis.get('stability', 0.0) < 0.7:
                recommendations.append("La estabilidad de transición es baja, considerar ajustar beta1 y beta2")
            
            if transition_analysis.get('transition_smoothness', 0.0) < 0.6:
                recommendations.append("La transición SGD-Adam no es suave, considerar aumentar warmup_steps")
            
            if metrics.convergence_speed < 0.01:
                recommendations.append("La convergencia es lenta, considerar aumentar la tasa de aprendizaje")
            
            if metrics.training_stability < 0.7:
                recommendations.append("La estabilidad del entrenamiento es baja, considerar reducir el learning rate")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones RAdam: {e}")
        
        return recommendations

class RAdam(torch.optim.Optimizer):
    """Implementación del optimizador RAdam"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)
        
        self.bias_correction_factor = 0.0
    
    def step(self, closure=None):
        """Paso de optimización RAdam"""
        loss = None
        if closure is not None:
            loss = closure()
        
        bias_correction_factors = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam no soporta gradientes dispersos')
                
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
                
                # RAdam: usar corrección de bias adaptativa
                if state['step'] > 1:
                    # Calcular factor de corrección
                    rho_inf = 2.0 / (1.0 - beta2) - 1.0
                    rho_t = rho_inf - 2.0 * state['step'] * beta2 ** state['step'] / (1.0 - beta2 ** state['step'])
                    
                    if rho_t > 4.0:  # Condición para usar Adam
                        # Usar Adam con corrección de bias
                        denom = (exp_avg_sq.div(bias_correction2).sqrt().add_(group['eps']))
                        step_size = group['lr'] / bias_correction1
                        bias_correction_factor = 1.0
                    else:
                        # Usar SGD (sin corrección de bias)
                        denom = torch.ones_like(exp_avg_sq)
                        step_size = group['lr']
                        bias_correction_factor = 0.0
                else:
                    # Primera iteración: usar SGD
                    denom = torch.ones_like(exp_avg_sq)
                    step_size = group['lr']
                    bias_correction_factor = 0.0
                
                # Aplicar actualización
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                bias_correction_factors.append(bias_correction_factor)
        
        if bias_correction_factors:
            self.bias_correction_factor = np.mean(bias_correction_factors)
        
        return loss

def create_radam_optimizer(config: AdvancedOptimizerConfig = None) -> RAdamWeightOptimizer:
    """Crea un optimizador RAdam"""
    return RAdamWeightOptimizer(config or AdvancedOptimizerConfig())

def analyze_radam_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                           criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de RAdam en un modelo"""
    try:
        optimizer = RAdamWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'transition_analysis': result.performance_analysis.get('transition_analysis', {})
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento RAdam: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_4.py - RAdam Optimizer cargado exitosamente")
