"""
RFEN8_RN_7.py - NovoGrad Optimizer
==================================

Implementación del optimizador NovoGrad desarrollado por NVIDIA para
entrenamiento eficiente de redes neuronales con mejor estabilidad.

Características principales:
- Normalización por capa de gradientes
- Mejor estabilidad numérica
- Convergencia más rápida
- Menor uso de memoria
- Optimización para GPU

Referencias:
- Ginsburg, B., et al. "Training Deep Neural Networks with Mixed Precision"
- Implementación basada en la documentación de NVIDIA
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

class NovoGradWeightOptimizer(BaseAdvancedOptimizer):
    """Optimizador NovoGrad con normalización por capa"""
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.novograd_beta1 = config.novograd_beta1
        self.novograd_beta2 = config.novograd_beta2
        self.novograd_eps = config.novograd_eps
        self.novograd_grad_averaging = config.novograd_grad_averaging
        self.gradient_norm_history = []
        self.layer_normalization_metrics = {}
        
        logger.info(f"NovoGradWeightOptimizer inicializado con beta1={self.novograd_beta1}, beta2={self.novograd_beta2}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador NovoGrad"""
        try:
            novograd_optimizer = NovoGrad(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.novograd_beta1, self.novograd_beta2),
                eps=self.novograd_eps,
                weight_decay=self.config.weight_decay,
                grad_averaging=self.novograd_grad_averaging
            )
            
            self.optimizer = novograd_optimizer
            logger.info("Optimizador NovoGrad creado exitosamente")
            return novograd_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador NovoGrad: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos del modelo usando NovoGrad"""
        try:
            logger.info("Iniciando optimización NovoGrad")
            start_time = time.time()
            
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            optimizer = self.create_optimizer(model)
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            
            model.train()
            loss_history = []
            accuracy_history = []
            gradient_norm_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    loss = self._novograd_step(model, data, target, optimizer, criterion)
                    epoch_losses.append(loss.item())
                    
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    if hasattr(optimizer, 'gradient_norm'):
                        gradient_norm_history.append(optimizer.gradient_norm)
                
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
            
            # Análisis de normalización por capa
            normalization_analysis = self._analyze_layer_normalization(gradient_norm_history)
            
            metrics = OptimizationMetrics(
                optimizer_name="NovoGrad",
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
                optimization_robustness=normalization_analysis['robustness'],
                overall_score=self._calculate_novograd_score(initial_metrics, final_metrics, normalization_analysis),
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
                performance_analysis={'normalization_analysis': normalization_analysis},
                recommendations=self._generate_novograd_recommendations(metrics, normalization_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización NovoGrad completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización NovoGrad: {e}")
            return OptimizationResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights=None,
                convergence_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(e)
            )
    
    def _novograd_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                       optimizer: 'NovoGrad', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización NovoGrad"""
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss
        except Exception as e:
            logger.error(f"Error en paso NovoGrad: {e}")
            raise
    
    def _analyze_layer_normalization(self, gradient_norm_history: List[float]) -> Dict:
        """Analiza la normalización por capa"""
        try:
            if not gradient_norm_history:
                return {'robustness': 0.0, 'normalization_effectiveness': 0.0, 'stability': 0.0}
            
            # Calcular efectividad de normalización
            mean_norm = np.mean(gradient_norm_history)
            std_norm = np.std(gradient_norm_history)
            normalization_effectiveness = max(0.0, 1.0 - std_norm / max(mean_norm, 1e-8))
            
            # Calcular estabilidad
            stability = max(0.0, 1.0 - std_norm / max(mean_norm, 1e-8))
            
            # Calcular robustez
            robustness = (normalization_effectiveness + stability) / 2.0
            
            return {
                'robustness': robustness,
                'normalization_effectiveness': normalization_effectiveness,
                'stability': stability,
                'mean_gradient_norm': mean_norm,
                'gradient_norm_variance': std_norm
            }
            
        except Exception as e:
            logger.error(f"Error analizando normalización por capa: {e}")
            return {'robustness': 0.0, 'normalization_effectiveness': 0.0, 'stability': 0.0}
    
    def _calculate_novograd_score(self, initial_metrics: Dict, final_metrics: Dict, 
                                  normalization_analysis: Dict) -> float:
        """Calcula el score específico de NovoGrad"""
        try:
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            robustness = normalization_analysis.get('robustness', 0.0)
            normalization_effectiveness = normalization_analysis.get('normalization_effectiveness', 0.0)
            
            novograd_score = (
                loss_improvement * 0.3 +
                accuracy_improvement * 0.3 +
                robustness * 0.2 +
                normalization_effectiveness * 0.2
            )
            
            return max(0.0, min(1.0, novograd_score))
            
        except Exception as e:
            logger.error(f"Error calculando score NovoGrad: {e}")
            return 0.0
    
    def _generate_novograd_recommendations(self, metrics: OptimizationMetrics, 
                                          normalization_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para NovoGrad"""
        recommendations = []
        
        try:
            if normalization_analysis.get('normalization_effectiveness', 0.0) < 0.7:
                recommendations.append("La efectividad de normalización es baja, considerar ajustar beta1 y beta2")
            
            if normalization_analysis.get('stability', 0.0) < 0.6:
                recommendations.append("La estabilidad de normalización es baja, considerar ajustar eps")
            
            if metrics.convergence_speed < 0.01:
                recommendations.append("La convergencia es lenta, considerar aumentar la tasa de aprendizaje")
            
            if metrics.training_stability < 0.7:
                recommendations.append("La estabilidad del entrenamiento es baja, considerar reducir el learning rate")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones NovoGrad: {e}")
        
        return recommendations

class NovoGrad(torch.optim.Optimizer):
    """Implementación del optimizador NovoGrad"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, grad_averaging=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging)
        super(NovoGrad, self).__init__(params, defaults)
        
        self.gradient_norm = 0.0
    
    def step(self, closure=None):
        """Paso de optimización NovoGrad"""
        loss = None
        if closure is not None:
            loss = closure()
        
        gradient_norms = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NovoGrad no soporta gradientes dispersos')
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # NovoGrad: normalización por capa
                grad_norm = torch.norm(grad)
                gradient_norms.append(grad_norm.item())
                
                # Normalizar gradiente
                if grad_norm > 0:
                    grad_normalized = grad / grad_norm
                else:
                    grad_normalized = grad
                
                # Actualizar estimaciones
                exp_avg.mul_(beta1).add_(grad_normalized, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_normalized, grad_normalized, value=1 - beta2)
                
                # Calcular bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Calcular paso de actualización
                denom = (exp_avg_sq.div(bias_correction2).sqrt().add_(group['eps']))
                step_size = group['lr'] / bias_correction1
                
                # Aplicar actualización
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Aplicar weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
        
        if gradient_norms:
            self.gradient_norm = np.mean(gradient_norms)
        
        return loss

def create_novograd_optimizer(config: AdvancedOptimizerConfig = None) -> NovoGradWeightOptimizer:
    """Crea un optimizador NovoGrad"""
    return NovoGradWeightOptimizer(config or AdvancedOptimizerConfig())

def analyze_novograd_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de NovoGrad en un modelo"""
    try:
        optimizer = NovoGradWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'normalization_analysis': result.performance_analysis.get('normalization_analysis', {})
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento NovoGrad: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_7.py - NovoGrad Optimizer cargado exitosamente")
