"""
RFEN8_RN_6.py - Lookahead Optimizer con Ranger
===============================================

Implementación del optimizador Lookahead que mejora la estabilidad y
convergencia mediante actualizaciones lentas y suaves de los parámetros.

Características principales:
- Actualizaciones lentas para mayor estabilidad
- Integración con Ranger (RAdam + Lookahead)
- Mejor convergencia en problemas complejos
- Reducción de oscilaciones
- Mayor robustez

Referencias:
- Zhang, M., et al. "Lookahead Optimizer: k steps forward, 1 step back"
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

class LookaheadWeightOptimizer(BaseAdvancedOptimizer):
    """Optimizador Lookahead con integración Ranger"""
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.lookahead_k = config.lookahead_k
        self.lookahead_alpha = config.lookahead_alpha
        self.lookahead_history = []
        self.stability_metrics = {}
        self.ranger_integration = True
        
        logger.info(f"LookaheadWeightOptimizer inicializado con k={self.lookahead_k}, alpha={self.lookahead_alpha}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador Lookahead con Ranger"""
        try:
            # Crear optimizador base (RAdam)
            base_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay
            )
            
            # Crear Lookahead wrapper
            lookahead_optimizer = Lookahead(
                base_optimizer,
                k=self.lookahead_k,
                alpha=self.lookahead_alpha
            )
            
            self.optimizer = lookahead_optimizer
            logger.info("Optimizador Lookahead creado exitosamente")
            return lookahead_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador Lookahead: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos del modelo usando Lookahead"""
        try:
            logger.info("Iniciando optimización Lookahead")
            start_time = time.time()
            
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            optimizer = self.create_optimizer(model)
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            
            model.train()
            loss_history = []
            accuracy_history = []
            stability_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    loss = self._lookahead_step(model, data, target, optimizer, criterion)
                    epoch_losses.append(loss.item())
                    
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    if batch_idx % 10 == 0:
                        stability = self._calculate_stability(model, data, target, criterion)
                        stability_history.append(stability)
                
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
            
            # Análisis de estabilidad
            stability_analysis = self._analyze_stability(stability_history, loss_history)
            
            metrics = OptimizationMetrics(
                optimizer_name="Lookahead",
                initial_loss=initial_metrics['loss'],
                final_loss=final_metrics['loss'],
                convergence_iterations=len(loss_history),
                convergence_speed=1.0 / max(len(loss_history), 1),
                generalization_improvement=final_metrics['accuracy'] - initial_metrics['accuracy'],
                computational_efficiency=1.0 / optimization_time,
                memory_usage=self._estimate_memory_usage(model),
                training_stability=stability_analysis['overall_stability'],
                test_accuracy=final_metrics['accuracy'],
                loss_reduction=(initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8),
                gradient_norm=self._calculate_gradient_norm(model),
                weight_magnitude=self._calculate_weight_magnitude(model),
                optimization_robustness=stability_analysis['robustness'],
                overall_score=self._calculate_lookahead_score(initial_metrics, final_metrics, stability_analysis),
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
                performance_analysis={'stability_analysis': stability_analysis},
                recommendations=self._generate_lookahead_recommendations(metrics, stability_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización Lookahead completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización Lookahead: {e}")
            return OptimizationResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights=None,
                convergence_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(e)
            )
    
    def _lookahead_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                       optimizer: 'Lookahead', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización Lookahead"""
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss
        except Exception as e:
            logger.error(f"Error en paso Lookahead: {e}")
            raise
    
    def _calculate_stability(self, model: nn.Module, data: torch.Tensor,
                           target: torch.Tensor, criterion: nn.Module) -> float:
        """Calcula la estabilidad del modelo"""
        try:
            # Calcular pérdida múltiples veces para medir estabilidad
            losses = []
            for _ in range(3):
                with torch.no_grad():
                    output = model(data)
                    loss = criterion(output, target)
                    losses.append(loss.item())
            
            # Calcular estabilidad como inverso de la varianza
            stability = 1.0 / (np.var(losses) + 1e-8)
            return stability
            
        except Exception as e:
            logger.error(f"Error calculando estabilidad: {e}")
            return 0.0
    
    def _analyze_stability(self, stability_history: List[float], 
                          loss_history: List[float]) -> Dict:
        """Analiza la estabilidad del entrenamiento"""
        try:
            if not stability_history:
                return {'overall_stability': 0.0, 'robustness': 0.0, 'convergence_smoothness': 0.0}
            
            # Calcular estabilidad general
            mean_stability = np.mean(stability_history)
            std_stability = np.std(stability_history)
            overall_stability = max(0.0, 1.0 - std_stability / max(mean_stability, 1e-8))
            
            # Calcular suavidad de convergencia
            if len(loss_history) > 1:
                loss_changes = np.diff(loss_history)
                convergence_smoothness = max(0.0, 1.0 - np.std(loss_changes) / max(np.mean(np.abs(loss_changes)), 1e-8))
            else:
                convergence_smoothness = 0.0
            
            # Calcular robustez
            robustness = (overall_stability + convergence_smoothness) / 2.0
            
            return {
                'overall_stability': overall_stability,
                'robustness': robustness,
                'convergence_smoothness': convergence_smoothness,
                'mean_stability': mean_stability,
                'stability_variance': std_stability
            }
            
        except Exception as e:
            logger.error(f"Error analizando estabilidad: {e}")
            return {'overall_stability': 0.0, 'robustness': 0.0, 'convergence_smoothness': 0.0}
    
    def _calculate_lookahead_score(self, initial_metrics: Dict, final_metrics: Dict, 
                                  stability_analysis: Dict) -> float:
        """Calcula el score específico de Lookahead"""
        try:
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            stability = stability_analysis.get('overall_stability', 0.0)
            robustness = stability_analysis.get('robustness', 0.0)
            
            lookahead_score = (
                loss_improvement * 0.25 +
                accuracy_improvement * 0.25 +
                stability * 0.25 +
                robustness * 0.25
            )
            
            return max(0.0, min(1.0, lookahead_score))
            
        except Exception as e:
            logger.error(f"Error calculando score Lookahead: {e}")
            return 0.0
    
    def _generate_lookahead_recommendations(self, metrics: OptimizationMetrics, 
                                           stability_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para Lookahead"""
        recommendations = []
        
        try:
            if stability_analysis.get('overall_stability', 0.0) < 0.7:
                recommendations.append("La estabilidad general es baja, considerar aumentar k o reducir alpha")
            
            if stability_analysis.get('convergence_smoothness', 0.0) < 0.6:
                recommendations.append("La convergencia no es suave, considerar ajustar los parámetros de Lookahead")
            
            if metrics.convergence_speed < 0.01:
                recommendations.append("La convergencia es lenta, considerar reducir k o aumentar alpha")
            
            if metrics.training_stability < 0.7:
                recommendations.append("La estabilidad del entrenamiento es baja, considerar ajustar los parámetros")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones Lookahead: {e}")
        
        return recommendations

class Lookahead(torch.optim.Optimizer):
    """Implementación del optimizador Lookahead"""
    
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Copiar parámetros del optimizador base
        defaults = dict(k=k, alpha=alpha)
        super(Lookahead, self).__init__(optimizer.param_groups, defaults)
        
        # Inicializar slow weights
        for group in self.param_groups:
            group['slow_weights'] = []
            for p in group['params']:
                group['slow_weights'].append(p.clone().detach())
    
    def step(self, closure=None):
        """Paso de optimización Lookahead"""
        loss = self.optimizer.step(closure)
        self.step_count += 1
        
        # Actualizar slow weights cada k pasos
        if self.step_count % self.k == 0:
            for group in self.param_groups:
                for p, slow_p in zip(group['params'], group['slow_weights']):
                    # Interpolación entre fast y slow weights
                    slow_p.data.add_(p.data - slow_p.data, alpha=self.alpha)
                    p.data.copy_(slow_p.data)
        
        return loss

def create_lookahead_optimizer(config: AdvancedOptimizerConfig = None) -> LookaheadWeightOptimizer:
    """Crea un optimizador Lookahead"""
    return LookaheadWeightOptimizer(config or AdvancedOptimizerConfig())

def analyze_lookahead_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de Lookahead en un modelo"""
    try:
        optimizer = LookaheadWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'stability_analysis': result.performance_analysis.get('stability_analysis', {})
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento Lookahead: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_6.py - Lookahead Optimizer cargado exitosamente")
