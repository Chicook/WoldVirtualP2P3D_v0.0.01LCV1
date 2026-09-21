"""
RFEN8_RN_2.py - Lion Optimizer (Google 2023)
============================================

Implementación del optimizador Lion (EvoLved Sign Momentum) desarrollado
por Google en 2023. Lion es un optimizador eficiente que combina la
simplicidad de SGD con la adaptabilidad de Adam.

Características principales:
- Optimización eficiente con memoria limitada
- Uso de signos de gradientes para actualizaciones
- Combinación de momentum y adaptación
- Mejor rendimiento en tareas de lenguaje y visión
- Menor uso de memoria comparado con Adam

Referencias:
- Chen, X., et al. "Symbolic Discovery of Optimization Algorithms"
- Implementación basada en el paper original de Google
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass  # dependencia pesada opcional
try:
    torch
except NameError:
    import types as _t
    torch = _t.SimpleNamespace(
        no_grad=lambda *a, **k: (lambda f: f) if a and callable(a[0]) else (lambda f: f),
        optim=_t.SimpleNamespace(Optimizer=object),
        Tensor=object,
    )
try:
    nn
except NameError:
    import types as _t2
    nn = _t2.SimpleNamespace(Module=object)
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import time
import copy
from ..RFENRN8 import BaseAdvancedOptimizer, AdvancedOptimizerConfig, OptimizationResult, OptimizationMetrics

logger = logging.getLogger(__name__)

class LionWeightOptimizer(BaseAdvancedOptimizer):
    """
    Optimizador Lion (EvoLved Sign Momentum) para optimización eficiente
    de pesos neuronales con memoria limitada.
    """
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.lion_beta1 = config.lion_beta1
        self.lion_beta2 = config.lion_beta2
        self.momentum_history = []
        self.sign_history = []
        self.efficiency_metrics = {}
        self.memory_usage_tracker = []
        self.adaptive_lr = True
        self.lr_decay_factor = 0.99
        self.min_lr = 1e-6
        self.current_lr = config.learning_rate
        
        logger.info(f"LionWeightOptimizer inicializado con beta1={self.lion_beta1}, beta2={self.lion_beta2}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador Lion"""
        try:
            # Crear optimizador Lion personalizado
            lion_optimizer = Lion(
                model.parameters(),
                lr=self.current_lr,
                betas=(self.lion_beta1, self.lion_beta2),
                weight_decay=self.config.weight_decay
            )
            
            self.optimizer = lion_optimizer
            logger.info("Optimizador Lion creado exitosamente")
            return lion_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador Lion: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos del modelo usando Lion"""
        try:
            logger.info("Iniciando optimización Lion")
            start_time = time.time()
            
            # Configurar criterio de pérdida
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            # Crear optimizador Lion
            optimizer = self.create_optimizer(model)
            
            # Métricas iniciales
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            initial_loss = initial_metrics['loss']
            initial_accuracy = initial_metrics['accuracy']
            initial_memory = self._estimate_memory_usage(model)
            
            # Entrenamiento Lion
            model.train()
            loss_history = []
            accuracy_history = []
            memory_history = []
            efficiency_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                epoch_memory = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    # Paso Lion
                    loss = self._lion_step(model, data, target, optimizer, criterion)
                    
                    epoch_losses.append(loss.item())
                    
                    # Calcular precisión
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    # Monitorear memoria
                    current_memory = self._estimate_memory_usage(model)
                    epoch_memory.append(current_memory)
                    
                    # Calcular eficiencia
                    if batch_idx % 10 == 0:
                        efficiency = self._calculate_efficiency(model, data, target, criterion)
                        efficiency_history.append(efficiency)
                    
                    # Actualizar learning rate adaptativo
                    if self.adaptive_lr and batch_idx % 100 == 0:
                        self._update_adaptive_lr(loss.item(), loss_history)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = self.current_lr
                
                # Métricas de época
                avg_loss = np.mean(epoch_losses)
                avg_accuracy = np.mean(epoch_accuracies)
                avg_memory = np.mean(epoch_memory)
                
                loss_history.append(avg_loss)
                accuracy_history.append(avg_accuracy)
                memory_history.append(avg_memory)
                
                # Verificar convergencia
                if self._check_convergence(loss_history):
                    logger.info(f"Convergencia alcanzada en época {epoch}")
                    break
                
                # Logging
                if epoch % 10 == 0:
                    logger.info(f"Época {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}, Memory={avg_memory:.2f}MB")
            
            # Métricas finales
            final_metrics = self._evaluate_model(model, data_loader, criterion)
            final_loss = final_metrics['loss']
            final_accuracy = final_metrics['accuracy']
            final_memory = self._estimate_memory_usage(model)
            
            # Calcular métricas de optimización
            optimization_time = time.time() - start_time
            convergence_iterations = len(loss_history)
            convergence_speed = 1.0 / max(convergence_iterations, 1)
            generalization_improvement = final_accuracy - initial_accuracy
            loss_reduction = (initial_loss - final_loss) / max(initial_loss, 1e-8)
            memory_efficiency = (initial_memory - final_memory) / max(initial_memory, 1e-8)
            
            # Análisis de eficiencia
            efficiency_analysis = self._analyze_efficiency(efficiency_history, memory_history)
            
            # Crear métricas
            metrics = OptimizationMetrics(
                optimizer_name="Lion",
                initial_loss=initial_loss,
                final_loss=final_loss,
                convergence_iterations=convergence_iterations,
                convergence_speed=convergence_speed,
                generalization_improvement=generalization_improvement,
                computational_efficiency=efficiency_analysis['computational_efficiency'],
                memory_usage=final_memory,
                training_stability=1.0 - np.std(loss_history[-10:]) / max(np.mean(loss_history[-10:]), 1e-8),
                test_accuracy=final_accuracy,
                loss_reduction=loss_reduction,
                gradient_norm=self._calculate_gradient_norm(model),
                weight_magnitude=self._calculate_weight_magnitude(model),
                optimization_robustness=efficiency_analysis['robustness'],
                overall_score=self._calculate_lion_score(initial_metrics, final_metrics, efficiency_analysis),
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
                    'efficiency_analysis': efficiency_analysis,
                    'memory_analysis': self._analyze_memory_usage(memory_history),
                    'lion_metrics': self.efficiency_metrics
                },
                recommendations=self._generate_lion_recommendations(metrics, efficiency_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización Lion completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización Lion: {e}")
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
    
    def _lion_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                   optimizer: 'Lion', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización Lion"""
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Paso Lion
            optimizer.step()
            
            # Registrar métricas de Lion
            if hasattr(optimizer, 'momentum_magnitude'):
                self.momentum_history.append(optimizer.momentum_magnitude)
            
            if hasattr(optimizer, 'sign_ratio'):
                self.sign_history.append(optimizer.sign_ratio)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error en paso Lion: {e}")
            raise
    
    def _calculate_efficiency(self, model: nn.Module, data: torch.Tensor,
                             target: torch.Tensor, criterion: nn.Module) -> float:
        """Calcula la eficiencia computacional del modelo"""
        try:
            # Medir tiempo de forward pass
            start_time = time.time()
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
            forward_time = time.time() - start_time
            
            # Calcular eficiencia (inversamente proporcional al tiempo)
            efficiency = 1.0 / max(forward_time, 1e-6)
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error calculando eficiencia: {e}")
            return 0.0
    
    def _update_adaptive_lr(self, current_loss: float, loss_history: List[float]) -> None:
        """Actualiza el learning rate adaptativamente"""
        try:
            if len(loss_history) < 5:
                return
            
            # Calcular tendencia de pérdida
            recent_losses = loss_history[-5:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            # Ajustar learning rate basado en la tendencia
            if loss_trend < -0.01:  # Mejorando rápidamente
                self.current_lr *= 1.01
            elif loss_trend > 0.01:  # Empeorando
                self.current_lr *= self.lr_decay_factor
            
            # Mantener límites
            self.current_lr = max(self.min_lr, min(self.current_lr, self.config.learning_rate * 2))
            
        except Exception as e:
            logger.error(f"Error actualizando learning rate adaptativo: {e}")
    
    def _analyze_efficiency(self, efficiency_history: List[float], 
                           memory_history: List[float]) -> Dict:
        """Analiza la eficiencia computacional y de memoria"""
        try:
            if not efficiency_history:
                return {'computational_efficiency': 0.0, 'memory_efficiency': 0.0, 'robustness': 0.0}
            
            # Métricas de eficiencia computacional
            mean_efficiency = np.mean(efficiency_history)
            std_efficiency = np.std(efficiency_history)
            
            # Métricas de eficiencia de memoria
            if memory_history:
                memory_reduction = (memory_history[0] - memory_history[-1]) / max(memory_history[0], 1e-8)
                memory_stability = 1.0 - np.std(memory_history) / max(np.mean(memory_history), 1e-8)
            else:
                memory_reduction = 0.0
                memory_stability = 0.0
            
            # Calcular robustez
            robustness = max(0.0, 1.0 - std_efficiency / max(mean_efficiency, 1e-8))
            
            return {
                'computational_efficiency': mean_efficiency,
                'memory_efficiency': memory_reduction,
                'memory_stability': memory_stability,
                'robustness': robustness,
                'efficiency_trend': 'stable' if std_efficiency < mean_efficiency * 0.1 else 'variable'
            }
            
        except Exception as e:
            logger.error(f"Error analizando eficiencia: {e}")
            return {'computational_efficiency': 0.0, 'memory_efficiency': 0.0, 'robustness': 0.0}
    
    def _analyze_memory_usage(self, memory_history: List[float]) -> Dict:
        """Analiza el uso de memoria durante el entrenamiento"""
        try:
            if not memory_history:
                return {'peak_memory': 0.0, 'average_memory': 0.0, 'memory_trend': 'stable'}
            
            peak_memory = max(memory_history)
            average_memory = np.mean(memory_history)
            
            # Calcular tendencia de memoria
            if len(memory_history) > 1:
                memory_trend = np.polyfit(range(len(memory_history)), memory_history, 1)[0]
                if memory_trend > 0.1:
                    trend_str = 'increasing'
                elif memory_trend < -0.1:
                    trend_str = 'decreasing'
                else:
                    trend_str = 'stable'
            else:
                trend_str = 'stable'
            
            return {
                'peak_memory': peak_memory,
                'average_memory': average_memory,
                'memory_trend': trend_str,
                'memory_variance': np.std(memory_history)
            }
            
        except Exception as e:
            logger.error(f"Error analizando uso de memoria: {e}")
            return {'peak_memory': 0.0, 'average_memory': 0.0, 'memory_trend': 'stable'}
    
    def _calculate_lion_score(self, initial_metrics: Dict, final_metrics: Dict, 
                             efficiency_analysis: Dict) -> float:
        """Calcula el score específico de Lion"""
        try:
            # Mejora de pérdida
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            
            # Mejora de precisión
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            
            # Eficiencia computacional
            computational_efficiency = efficiency_analysis.get('computational_efficiency', 0.0)
            
            # Eficiencia de memoria
            memory_efficiency = efficiency_analysis.get('memory_efficiency', 0.0)
            
            # Score combinado
            lion_score = (
                loss_improvement * 0.25 +
                accuracy_improvement * 0.25 +
                computational_efficiency * 0.25 +
                memory_efficiency * 0.25
            )
            
            return max(0.0, min(1.0, lion_score))
            
        except Exception as e:
            logger.error(f"Error calculando score Lion: {e}")
            return 0.0
    
    def _generate_lion_recommendations(self, metrics: OptimizationMetrics, 
                                      efficiency_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para Lion"""
        recommendations = []
        
        try:
            # Recomendaciones basadas en eficiencia
            if efficiency_analysis.get('computational_efficiency', 0.0) < 0.5:
                recommendations.append("La eficiencia computacional es baja, considerar optimizar la arquitectura del modelo")
            
            if efficiency_analysis.get('memory_efficiency', 0.0) < 0.1:
                recommendations.append("El uso de memoria es alto, considerar técnicas de compresión de modelo")
            
            # Recomendaciones basadas en convergencia
            if metrics.convergence_speed < 0.01:
                recommendations.append("La convergencia es lenta, considerar ajustar los parámetros beta de Lion")
            
            # Recomendaciones basadas en estabilidad
            if metrics.training_stability < 0.7:
                recommendations.append("La estabilidad del entrenamiento es baja, considerar reducir el learning rate")
            
            # Recomendaciones específicas de Lion
            if len(self.momentum_history) > 0:
                avg_momentum = np.mean(self.momentum_history)
                if avg_momentum < 0.1:
                    recommendations.append("El momentum es muy bajo, considerar aumentar beta1")
                elif avg_momentum > 0.9:
                    recommendations.append("El momentum es muy alto, considerar reducir beta1")
            
            if len(self.sign_history) > 0:
                avg_sign_ratio = np.mean(self.sign_history)
                if avg_sign_ratio < 0.3:
                    recommendations.append("La proporción de signos es baja, considerar ajustar beta2")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones Lion: {e}")
        
        return recommendations

class Lion(torch.optim.Optimizer):
    """
    Implementación del optimizador Lion (EvoLved Sign Momentum)
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)
        
        self.momentum_magnitude = 0.0
        self.sign_ratio = 0.0
    
    def step(self, closure=None):
        """Paso de optimización Lion"""
        loss = None
        if closure is not None:
            loss = closure()
        
        momentum_magnitudes = []
        sign_counts = []
        total_params = 0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lion no soporta gradientes dispersos')
                
                state = self.state[p]
                
                # Estado inicial
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Actualización Lion
                update = exp_avg * beta1 + grad * (1 - beta1)
                update_sign = torch.sign(update)
                
                # Actualizar momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                # Aplicar actualización con signos
                p.data.mul_(1 - group['lr']).add_(update_sign, alpha=group['lr'])
                
                # Aplicar weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Registrar métricas
                momentum_magnitudes.append(torch.norm(exp_avg).item())
                sign_counts.append(torch.sum(update_sign != 0).item())
                total_params += update_sign.numel()
        
        # Calcular métricas globales
        if momentum_magnitudes:
            self.momentum_magnitude = np.mean(momentum_magnitudes)
        
        if sign_counts and total_params > 0:
            self.sign_ratio = sum(sign_counts) / total_params
        
        return loss

# Funciones de utilidad
def create_lion_optimizer(config: AdvancedOptimizerConfig = None) -> LionWeightOptimizer:
    """Crea un optimizador Lion"""
    return LionWeightOptimizer(config or AdvancedOptimizerConfig())

def analyze_lion_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                           criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de Lion en un modelo"""
    try:
        optimizer = LionWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'efficiency_analysis': result.performance_analysis.get('efficiency_analysis', {})
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento Lion: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_2.py - Lion Optimizer cargado exitosamente")
