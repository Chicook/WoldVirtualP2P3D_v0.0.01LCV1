"""
RFEN8_RN_3.py - AdaBelief Optimizer Avanzado
=============================================

Implementación del optimizador AdaBelief que adapta los pasos de aprendizaje
basándose en la creencia en las observaciones de gradientes, mejorando la
estabilidad y convergencia comparado con Adam.

Características principales:
- Adaptación basada en la creencia en gradientes
- Mejor manejo de gradientes ruidosos
- Convergencia más rápida y estable
- Menor sensibilidad a hiperparámetros
- Soporte para diferentes estrategias de adaptación

Referencias:
- Zhuang, J., et al. "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"
- Implementación basada en el paper original
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import time
import copy
from ..RFENRN8 import BaseAdvancedOptimizer, AdvancedOptimizerConfig, OptimizationResult, OptimizationMetrics

logger = logging.getLogger(__name__)

class AdaBeliefWeightOptimizer(BaseAdvancedOptimizer):
    """
    Optimizador AdaBelief que adapta los pasos de aprendizaje basándose
    en la creencia en las observaciones de gradientes.
    """
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.adabelief_beta1 = config.adabelief_beta1
        self.adabelief_beta2 = config.adabelief_beta2
        self.adabelief_eps = config.adabelief_eps
        self.belief_history = []
        self.gradient_belief_ratio = []
        self.adaptation_metrics = {}
        self.noise_robustness_tracker = []
        self.adaptive_eps = True
        self.eps_decay_factor = 0.99
        self.min_eps = 1e-8
        self.current_eps = self.adabelief_eps
        
        logger.info(f"AdaBeliefWeightOptimizer inicializado con beta1={self.adabelief_beta1}, beta2={self.adabelief_beta2}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador AdaBelief"""
        try:
            # Crear optimizador AdaBelief personalizado
            adabelief_optimizer = AdaBelief(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.adabelief_beta1, self.adabelief_beta2),
                eps=self.current_eps,
                weight_decay=self.config.weight_decay
            )
            
            self.optimizer = adabelief_optimizer
            logger.info("Optimizador AdaBelief creado exitosamente")
            return adabelief_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador AdaBelief: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos del modelo usando AdaBelief"""
        try:
            logger.info("Iniciando optimización AdaBelief")
            start_time = time.time()
            
            # Configurar criterio de pérdida
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            # Crear optimizador AdaBelief
            optimizer = self.create_optimizer(model)
            
            # Métricas iniciales
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            initial_loss = initial_metrics['loss']
            initial_accuracy = initial_metrics['accuracy']
            
            # Entrenamiento AdaBelief
            model.train()
            loss_history = []
            accuracy_history = []
            belief_history = []
            adaptation_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                epoch_beliefs = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    # Paso AdaBelief
                    loss = self._adabelief_step(model, data, target, optimizer, criterion)
                    
                    epoch_losses.append(loss.item())
                    
                    # Calcular precisión
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    # Registrar métricas de creencia
                    if hasattr(optimizer, 'belief_magnitude'):
                        epoch_beliefs.append(optimizer.belief_magnitude)
                    
                    # Calcular adaptación
                    if batch_idx % 10 == 0:
                        adaptation = self._calculate_adaptation(model, data, target, criterion)
                        adaptation_history.append(adaptation)
                    
                    # Actualizar epsilon adaptativo
                    if self.adaptive_eps and batch_idx % 50 == 0:
                        self._update_adaptive_eps(loss.item(), loss_history)
                        for param_group in optimizer.param_groups:
                            param_group['eps'] = self.current_eps
                
                # Métricas de época
                avg_loss = np.mean(epoch_losses)
                avg_accuracy = np.mean(epoch_accuracies)
                avg_belief = np.mean(epoch_beliefs) if epoch_beliefs else 0.0
                
                loss_history.append(avg_loss)
                accuracy_history.append(avg_accuracy)
                belief_history.append(avg_belief)
                
                # Verificar convergencia
                if self._check_convergence(loss_history):
                    logger.info(f"Convergencia alcanzada en época {epoch}")
                    break
                
                # Logging
                if epoch % 10 == 0:
                    logger.info(f"Época {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}, Belief={avg_belief:.4f}")
            
            # Métricas finales
            final_metrics = self._evaluate_model(model, data_loader, criterion)
            final_loss = final_metrics['loss']
            final_accuracy = final_metrics['accuracy']
            
            # Calcular métricas de optimización
            optimization_time = time.time() - start_time
            convergence_iterations = len(loss_history)
            convergence_speed = 1.0 / max(convergence_iterations, 1)
            generalization_improvement = final_accuracy - initial_accuracy
            loss_reduction = (initial_loss - final_loss) / max(initial_loss, 1e-8)
            
            # Análisis de adaptación y creencia
            adaptation_analysis = self._analyze_adaptation(adaptation_history, belief_history)
            
            # Crear métricas
            metrics = OptimizationMetrics(
                optimizer_name="AdaBelief",
                initial_loss=initial_loss,
                final_loss=final_loss,
                convergence_iterations=convergence_iterations,
                convergence_speed=convergence_speed,
                generalization_improvement=generalization_improvement,
                computational_efficiency=convergence_speed / optimization_time,
                memory_usage=self._estimate_memory_usage(model),
                training_stability=1.0 - np.std(loss_history[-10:]) / max(np.mean(loss_history[-10:]), 1e-8),
                test_accuracy=final_accuracy,
                loss_reduction=loss_reduction,
                gradient_norm=self._calculate_gradient_norm(model),
                weight_magnitude=self._calculate_weight_magnitude(model),
                optimization_robustness=adaptation_analysis['robustness'],
                overall_score=self._calculate_adabelief_score(initial_metrics, final_metrics, adaptation_analysis),
                optimization_time=optimization_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Crear resultado
            result = OptimizationResult(
                success=True,
                optimized_model=model,
                metrics=metrics,
                optimization_history=loss_history,
                best_weights=copy.deepcopy(model.state_dict()),
                convergence_analysis=self.analyze_convergence(loss_history),
                performance_analysis={
                    'adaptation_analysis': adaptation_analysis,
                    'belief_analysis': self._analyze_belief_patterns(belief_history),
                    'adabelief_metrics': self.adaptation_metrics
                },
                recommendations=self._generate_adabelief_recommendations(metrics, adaptation_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización AdaBelief completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización AdaBelief: {e}")
            return OptimizationResult(
                success=False,
                optimized_model=None,
                metrics=None,
                optimization_history=[],
                best_weights=None,
                convergence_analysis={},
                performance_analysis={},
                recommendations=[],
                error_message=str(e)
            )
    
    def _adabelief_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                        optimizer: 'AdaBelief', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización AdaBelief"""
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Paso AdaBelief
            optimizer.step()
            
            # Registrar métricas de creencia
            if hasattr(optimizer, 'belief_magnitude'):
                self.belief_history.append(optimizer.belief_magnitude)
            
            if hasattr(optimizer, 'gradient_belief_ratio'):
                self.gradient_belief_ratio.append(optimizer.gradient_belief_ratio)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error en paso AdaBelief: {e}")
            raise
    
    def _calculate_adaptation(self, model: nn.Module, data: torch.Tensor,
                            target: torch.Tensor, criterion: nn.Module) -> float:
        """Calcula el nivel de adaptación del optimizador"""
        try:
            # Calcular gradientes actuales
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Calcular norma de gradientes
            total_grad_norm = 0.0
            total_param_norm = 0.0
            
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += torch.norm(param.grad).item() ** 2
                    total_param_norm += torch.norm(param).item() ** 2
            
            total_grad_norm = np.sqrt(total_grad_norm)
            total_param_norm = np.sqrt(total_param_norm)
            
            # Calcular ratio de adaptación
            adaptation_ratio = total_grad_norm / max(total_param_norm, 1e-8)
            
            return adaptation_ratio
            
        except Exception as e:
            logger.error(f"Error calculando adaptación: {e}")
            return 0.0
    
    def _update_adaptive_eps(self, current_loss: float, loss_history: List[float]) -> None:
        """Actualiza epsilon adaptativamente basado en el progreso"""
        try:
            if len(loss_history) < 5:
                return
            
            # Calcular tendencia de pérdida
            recent_losses = loss_history[-5:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            # Ajustar epsilon basado en la tendencia
            if loss_trend < -0.01:  # Mejorando rápidamente
                self.current_eps *= 1.01
            elif loss_trend > 0.01:  # Empeorando
                self.current_eps *= self.eps_decay_factor
            
            # Mantener límites
            self.current_eps = max(self.min_eps, 
                                 min(self.current_eps, self.adabelief_eps * 10))
            
        except Exception as e:
            logger.error(f"Error actualizando epsilon adaptativo: {e}")
    
    def _analyze_adaptation(self, adaptation_history: List[float], 
                           belief_history: List[float]) -> Dict:
        """Analiza la adaptación y creencia del optimizador"""
        try:
            if not adaptation_history:
                return {'adaptation_stability': 0.0, 'belief_consistency': 0.0, 'robustness': 0.0}
            
            # Métricas de adaptación
            mean_adaptation = np.mean(adaptation_history)
            std_adaptation = np.std(adaptation_history)
            adaptation_stability = max(0.0, 1.0 - std_adaptation / max(mean_adaptation, 1e-8))
            
            # Métricas de creencia
            if belief_history:
                mean_belief = np.mean(belief_history)
                std_belief = np.std(belief_history)
                belief_consistency = max(0.0, 1.0 - std_belief / max(mean_belief, 1e-8))
            else:
                belief_consistency = 0.0
            
            # Calcular robustez general
            robustness = (adaptation_stability + belief_consistency) / 2.0
            
            # Calcular tendencia de adaptación
            if len(adaptation_history) > 1:
                adaptation_trend = np.polyfit(range(len(adaptation_history)), adaptation_history, 1)[0]
                if adaptation_trend < -0.001:
                    trend_str = 'decreasing'
                elif adaptation_trend > 0.001:
                    trend_str = 'increasing'
                else:
                    trend_str = 'stable'
            else:
                trend_str = 'stable'
            
            return {
                'mean_adaptation': mean_adaptation,
                'adaptation_stability': adaptation_stability,
                'belief_consistency': belief_consistency,
                'robustness': robustness,
                'adaptation_trend': trend_str,
                'final_adaptation': adaptation_history[-1] if adaptation_history else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analizando adaptación: {e}")
            return {'adaptation_stability': 0.0, 'belief_consistency': 0.0, 'robustness': 0.0}
    
    def _analyze_belief_patterns(self, belief_history: List[float]) -> Dict:
        """Analiza los patrones de creencia durante el entrenamiento"""
        try:
            if not belief_history:
                return {'belief_stability': 0.0, 'belief_trend': 'stable', 'noise_robustness': 0.0}
            
            # Calcular estabilidad de creencia
            mean_belief = np.mean(belief_history)
            std_belief = np.std(belief_history)
            belief_stability = max(0.0, 1.0 - std_belief / max(mean_belief, 1e-8))
            
            # Calcular tendencia de creencia
            if len(belief_history) > 1:
                belief_trend = np.polyfit(range(len(belief_history)), belief_history, 1)[0]
                if belief_trend < -0.001:
                    trend_str = 'decreasing'
                elif belief_trend > 0.001:
                    trend_str = 'increasing'
                else:
                    trend_str = 'stable'
            else:
                trend_str = 'stable'
            
            # Calcular robustez al ruido
            noise_robustness = max(0.0, 1.0 - std_belief / max(mean_belief, 1e-8))
            
            return {
                'mean_belief': mean_belief,
                'belief_stability': belief_stability,
                'belief_trend': trend_str,
                'noise_robustness': noise_robustness,
                'belief_variance': std_belief
            }
            
        except Exception as e:
            logger.error(f"Error analizando patrones de creencia: {e}")
            return {'belief_stability': 0.0, 'belief_trend': 'stable', 'noise_robustness': 0.0}
    
    def _calculate_adabelief_score(self, initial_metrics: Dict, final_metrics: Dict, 
                                  adaptation_analysis: Dict) -> float:
        """Calcula el score específico de AdaBelief"""
        try:
            # Mejora de pérdida
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            
            # Mejora de precisión
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            
            # Estabilidad de adaptación
            adaptation_stability = adaptation_analysis.get('adaptation_stability', 0.0)
            
            # Consistencia de creencia
            belief_consistency = adaptation_analysis.get('belief_consistency', 0.0)
            
            # Score combinado
            adabelief_score = (
                loss_improvement * 0.3 +
                accuracy_improvement * 0.3 +
                adaptation_stability * 0.2 +
                belief_consistency * 0.2
            )
            
            return max(0.0, min(1.0, adabelief_score))
            
        except Exception as e:
            logger.error(f"Error calculando score AdaBelief: {e}")
            return 0.0
    
    def _generate_adabelief_recommendations(self, metrics: OptimizationMetrics, 
                                           adaptation_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para AdaBelief"""
        recommendations = []
        
        try:
            # Recomendaciones basadas en adaptación
            if adaptation_analysis.get('adaptation_stability', 0.0) < 0.7:
                recommendations.append("La estabilidad de adaptación es baja, considerar ajustar beta1 y beta2")
            
            if adaptation_analysis.get('belief_consistency', 0.0) < 0.6:
                recommendations.append("La consistencia de creencia es baja, considerar aumentar beta2")
            
            # Recomendaciones basadas en convergencia
            if metrics.convergence_speed < 0.01:
                recommendations.append("La convergencia es lenta, considerar aumentar la tasa de aprendizaje")
            
            # Recomendaciones basadas en estabilidad
            if metrics.training_stability < 0.7:
                recommendations.append("La estabilidad del entrenamiento es baja, considerar reducir el learning rate")
            
            # Recomendaciones específicas de AdaBelief
            if self.current_eps < self.min_eps * 2:
                recommendations.append("El epsilon adaptativo es muy pequeño, considerar reinicializar")
            
            if len(self.belief_history) > 0:
                avg_belief = np.mean(self.belief_history)
                if avg_belief < 0.1:
                    recommendations.append("La creencia promedio es muy baja, considerar ajustar los parámetros beta")
                elif avg_belief > 0.9:
                    recommendations.append("La creencia promedio es muy alta, considerar reducir beta2")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones AdaBelief: {e}")
        
        return recommendations

class AdaBelief(torch.optim.Optimizer):
    """
    Implementación del optimizador AdaBelief
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaBelief, self).__init__(params, defaults)
        
        self.belief_magnitude = 0.0
        self.gradient_belief_ratio = 0.0
    
    def step(self, closure=None):
        """Paso de optimización AdaBelief"""
        loss = None
        if closure is not None:
            loss = closure()
        
        belief_magnitudes = []
        gradient_belief_ratios = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief no soporta gradientes dispersos')
                
                state = self.state[p]
                
                # Estado inicial
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Actualizar estimaciones de primer y segundo momento
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # AdaBelief: usar la diferencia entre gradiente y su promedio
                grad_diff = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_diff, grad_diff, value=1 - beta2)
                
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
                
                # Registrar métricas
                belief_magnitudes.append(torch.norm(exp_avg_sq).item())
                gradient_belief_ratios.append(torch.norm(grad_diff).item() / max(torch.norm(grad).item(), 1e-8))
        
        # Calcular métricas globales
        if belief_magnitudes:
            self.belief_magnitude = np.mean(belief_magnitudes)
        
        if gradient_belief_ratios:
            self.gradient_belief_ratio = np.mean(gradient_belief_ratios)
        
        return loss

# Funciones de utilidad
def create_adabelief_optimizer(config: AdvancedOptimizerConfig = None) -> AdaBeliefWeightOptimizer:
    """Crea un optimizador AdaBelief"""
    return AdaBeliefWeightOptimizer(config or AdvancedOptimizerConfig())

def analyze_adabelief_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                 criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de AdaBelief en un modelo"""
    try:
        optimizer = AdaBeliefWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'adaptation_analysis': result.performance_analysis.get('adaptation_analysis', {})
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento AdaBelief: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_3.py - AdaBelief Optimizer cargado exitosamente")
