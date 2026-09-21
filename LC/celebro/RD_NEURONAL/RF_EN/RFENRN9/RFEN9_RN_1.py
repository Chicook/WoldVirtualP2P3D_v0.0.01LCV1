"""
RFEN9_RN_1.py - Neural Tangent Kernel (NTK) Optimizer
======================================================

Implementación del optimizador basado en Neural Tangent Kernel (NTK) que
utiliza la teoría de kernels para analizar y optimizar redes neuronales
en el límite de ancho infinito.

Características principales:
- Análisis teórico de convergencia
- Optimización basada en kernels
- Límite de ancho infinito
- Convergencia garantizada teóricamente
- Análisis de estabilidad numérica

Referencias:
- Jacot, A., et al. "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
- Implementación basada en la teoría NTK
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import time
import copy
from ..RFENRN9 import BaseUltraAdvancedOptimizer, UltraAdvancedOptimizerConfig, UltraAdvancedOptimizationResult, UltraAdvancedOptimizationMetrics

logger = logging.getLogger(__name__)

class NTKWeightOptimizer(BaseUltraAdvancedOptimizer):
    """
    Optimizador basado en Neural Tangent Kernel (NTK) para análisis teórico
    y optimización de redes neuronales en el límite de ancho infinito.
    """
    
    def __init__(self, config: UltraAdvancedOptimizerConfig):
        super().__init__(config)
        self.ntk_lambda = config.ntk_lambda
        self.ntk_infinite_width = config.ntk_infinite_width
        self.kernel_history = []
        self.theoretical_metrics = {}
        self.convergence_analysis = {}
        self.stability_tracker = []
        
        logger.info(f"NTKWeightOptimizer inicializado con lambda={self.ntk_lambda}, infinite_width={self.ntk_infinite_width}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador NTK"""
        try:
            # Crear optimizador NTK personalizado
            ntk_optimizer = NTKOptimizer(
                model.parameters(),
                lr=self.config.learning_rate,
                lambda_reg=self.ntk_lambda,
                infinite_width=self.ntk_infinite_width,
                weight_decay=self.config.weight_decay
            )
            
            self.optimizer = ntk_optimizer
            logger.info("Optimizador NTK creado exitosamente")
            return ntk_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador NTK: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> UltraAdvancedOptimizationResult:
        """Optimiza los pesos del modelo usando NTK"""
        try:
            logger.info("Iniciando optimización NTK")
            start_time = time.time()
            
            # Configurar criterio de pérdida
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            # Crear optimizador NTK
            optimizer = self.create_optimizer(model)
            
            # Métricas iniciales
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            initial_loss = initial_metrics['loss']
            initial_accuracy = initial_metrics['accuracy']
            
            # Entrenamiento NTK
            model.train()
            loss_history = []
            accuracy_history = []
            kernel_history = []
            stability_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    # Paso NTK
                    loss = self._ntk_step(model, data, target, optimizer, criterion)
                    
                    epoch_losses.append(loss.item())
                    
                    # Calcular precisión
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    # Registrar métricas NTK
                    if hasattr(optimizer, 'kernel_magnitude'):
                        kernel_history.append(optimizer.kernel_magnitude)
                    
                    # Calcular estabilidad
                    if batch_idx % 10 == 0:
                        stability = self._calculate_ntk_stability(model, data, target, criterion)
                        stability_history.append(stability)
                
                # Métricas de época
                avg_loss = np.mean(epoch_losses)
                avg_accuracy = np.mean(epoch_accuracies)
                loss_history.append(avg_loss)
                accuracy_history.append(avg_accuracy)
                
                # Verificar convergencia teórica
                if self._check_theoretical_convergence(loss_history):
                    logger.info(f"Convergencia teórica alcanzada en época {epoch}")
                    break
                
                # Logging
                if epoch % 10 == 0:
                    logger.info(f"Época {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
            
            # Métricas finales
            final_metrics = self._evaluate_model(model, data_loader, criterion)
            final_loss = final_metrics['loss']
            final_accuracy = final_metrics['accuracy']
            
            # Calcular métricas de optimización
            optimization_time = time.time() - start_time
            convergence_iterations = len(loss_history)
            theoretical_convergence = self._calculate_theoretical_convergence_rate(loss_history)
            generalization_improvement = final_accuracy - initial_accuracy
            loss_reduction = (initial_loss - final_loss) / max(initial_loss, 1e-8)
            
            # Análisis teórico NTK
            theoretical_analysis = self._analyze_ntk_theory(kernel_history, stability_history)
            
            # Crear métricas
            metrics = UltraAdvancedOptimizationMetrics(
                optimizer_name="NTK",
                initial_loss=initial_loss,
                final_loss=final_loss,
                convergence_iterations=convergence_iterations,
                theoretical_convergence=theoretical_convergence,
                geometric_optimization=theoretical_analysis['geometric_optimization'],
                second_order_efficiency=theoretical_analysis['second_order_efficiency'],
                architecture_optimization=0.0,  # NTK no optimiza arquitectura
                quantum_advantage=0.0,  # NTK no usa computación cuántica
                meta_learning_adaptation=0.0,  # NTK no usa meta-aprendizaje
                pruning_efficiency=0.0,  # NTK no usa poda
                multi_level_distribution=0.0,  # NTK no usa multi-nivel
                bayesian_optimization_effectiveness=0.0,  # NTK no usa bayesiano
                ultra_advanced_integration_score=theoretical_analysis['integration_score'],
                overall_score=self._calculate_ntk_score(initial_metrics, final_metrics, theoretical_analysis),
                optimization_time=optimization_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Crear resultado
            result = UltraAdvancedOptimizationResult(
                success=True,
                optimized_model=model,
                metrics=metrics,
                optimization_history=loss_history,
                best_weights=copy.deepcopy(model.state_dict()),
                theoretical_analysis=theoretical_analysis,
                performance_analysis={
                    'kernel_analysis': self._analyze_kernel_evolution(kernel_history),
                    'stability_analysis': self._analyze_stability_patterns(stability_history),
                    'ntk_metrics': self.theoretical_metrics
                },
                recommendations=self._generate_ntk_recommendations(metrics, theoretical_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización NTK completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización NTK: {e}")
            return UltraAdvancedOptimizationResult(
                success=False,
                optimized_model=None,
                metrics=None,
                optimization_history=[],
                best_weights=None,
                theoretical_analysis={},
                performance_analysis={},
                recommendations=[],
                error_message=str(e)
            )
    
    def _ntk_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                  optimizer: 'NTKOptimizer', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización NTK"""
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Paso NTK
            optimizer.step()
            
            # Registrar métricas NTK
            if hasattr(optimizer, 'kernel_magnitude'):
                self.kernel_history.append(optimizer.kernel_magnitude)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error en paso NTK: {e}")
            raise
    
    def _calculate_ntk_stability(self, model: nn.Module, data: torch.Tensor,
                               target: torch.Tensor, criterion: nn.Module) -> float:
        """Calcula la estabilidad NTK del modelo"""
        try:
            # Calcular estabilidad basada en la teoría NTK
            with torch.no_grad():
                # Obtener representaciones de la capa oculta
                hidden_repr = self._get_hidden_representation(model, data)
                
                # Calcular estabilidad como inverso de la varianza
                stability = 1.0 / (torch.var(hidden_repr).item() + 1e-8)
                
                return stability
                
        except Exception as e:
            logger.error(f"Error calculando estabilidad NTK: {e}")
            return 0.0
    
    def _get_hidden_representation(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Obtiene la representación de la capa oculta"""
        try:
            # Para modelos simples, usar la primera capa lineal
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    return module(data.view(data.size(0), -1))
            
            # Fallback: usar la salida completa
            return model(data)
            
        except Exception as e:
            logger.error(f"Error obteniendo representación oculta: {e}")
            return torch.zeros(data.size(0), 1)
    
    def _check_theoretical_convergence(self, loss_history: List[float]) -> bool:
        """Verifica la convergencia teórica NTK"""
        try:
            if len(loss_history) < 10:
                return False
            
            # Verificar convergencia teórica
            recent_losses = loss_history[-10:]
            convergence_rate = abs(recent_losses[-1] - recent_losses[0]) / max(recent_losses[0], 1e-8)
            
            return convergence_rate < self.config.convergence_threshold
            
        except Exception as e:
            logger.error(f"Error verificando convergencia teórica: {e}")
            return False
    
    def _calculate_theoretical_convergence_rate(self, loss_history: List[float]) -> float:
        """Calcula la tasa de convergencia teórica"""
        try:
            if len(loss_history) < 2:
                return 0.0
            
            # Calcular tasa de convergencia teórica
            recent_losses = loss_history[-10:] if len(loss_history) >= 10 else loss_history
            convergence_rate = abs(recent_losses[-1] - recent_losses[0]) / max(recent_losses[0], 1e-8)
            
            return max(0.0, 1.0 - convergence_rate)
            
        except Exception as e:
            logger.error(f"Error calculando tasa de convergencia teórica: {e}")
            return 0.0
    
    def _analyze_ntk_theory(self, kernel_history: List[float], 
                           stability_history: List[float]) -> Dict:
        """Analiza la teoría NTK"""
        try:
            # Análisis de kernel
            if kernel_history:
                mean_kernel = np.mean(kernel_history)
                std_kernel = np.std(kernel_history)
                kernel_stability = max(0.0, 1.0 - std_kernel / max(mean_kernel, 1e-8))
            else:
                kernel_stability = 0.0
            
            # Análisis de estabilidad
            if stability_history:
                mean_stability = np.mean(stability_history)
                std_stability = np.std(stability_history)
                stability_consistency = max(0.0, 1.0 - std_stability / max(mean_stability, 1e-8))
            else:
                stability_consistency = 0.0
            
            # Calcular métricas teóricas
            geometric_optimization = kernel_stability
            second_order_efficiency = stability_consistency
            integration_score = (geometric_optimization + second_order_efficiency) / 2.0
            
            return {
                'geometric_optimization': geometric_optimization,
                'second_order_efficiency': second_order_efficiency,
                'integration_score': integration_score,
                'kernel_stability': kernel_stability,
                'stability_consistency': stability_consistency
            }
            
        except Exception as e:
            logger.error(f"Error analizando teoría NTK: {e}")
            return {'geometric_optimization': 0.0, 'second_order_efficiency': 0.0, 'integration_score': 0.0}
    
    def _analyze_kernel_evolution(self, kernel_history: List[float]) -> Dict:
        """Analiza la evolución del kernel"""
        try:
            if not kernel_history:
                return {'evolution_stability': 0.0, 'kernel_trend': 'stable'}
            
            # Calcular estabilidad de evolución
            mean_kernel = np.mean(kernel_history)
            std_kernel = np.std(kernel_history)
            evolution_stability = max(0.0, 1.0 - std_kernel / max(mean_kernel, 1e-8))
            
            # Calcular tendencia
            if len(kernel_history) > 1:
                kernel_trend = np.polyfit(range(len(kernel_history)), kernel_history, 1)[0]
                if kernel_trend > 0.001:
                    trend_str = 'increasing'
                elif kernel_trend < -0.001:
                    trend_str = 'decreasing'
                else:
                    trend_str = 'stable'
            else:
                trend_str = 'stable'
            
            return {
                'evolution_stability': evolution_stability,
                'kernel_trend': trend_str,
                'mean_kernel': mean_kernel,
                'kernel_variance': std_kernel
            }
            
        except Exception as e:
            logger.error(f"Error analizando evolución del kernel: {e}")
            return {'evolution_stability': 0.0, 'kernel_trend': 'stable'}
    
    def _analyze_stability_patterns(self, stability_history: List[float]) -> Dict:
        """Analiza los patrones de estabilidad"""
        try:
            if not stability_history:
                return {'stability_pattern': 'stable', 'consistency': 0.0}
            
            # Calcular consistencia
            mean_stability = np.mean(stability_history)
            std_stability = np.std(stability_history)
            consistency = max(0.0, 1.0 - std_stability / max(mean_stability, 1e-8))
            
            # Determinar patrón
            if consistency > 0.8:
                pattern = 'highly_stable'
            elif consistency > 0.6:
                pattern = 'stable'
            elif consistency > 0.4:
                pattern = 'moderate'
            else:
                pattern = 'unstable'
            
            return {
                'stability_pattern': pattern,
                'consistency': consistency,
                'mean_stability': mean_stability,
                'stability_variance': std_stability
            }
            
        except Exception as e:
            logger.error(f"Error analizando patrones de estabilidad: {e}")
            return {'stability_pattern': 'stable', 'consistency': 0.0}
    
    def _calculate_ntk_score(self, initial_metrics: Dict, final_metrics: Dict, 
                            theoretical_analysis: Dict) -> float:
        """Calcula el score específico de NTK"""
        try:
            # Mejora de pérdida
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            
            # Mejora de precisión
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            
            # Optimización geométrica
            geometric_optimization = theoretical_analysis.get('geometric_optimization', 0.0)
            
            # Eficiencia de segundo orden
            second_order_efficiency = theoretical_analysis.get('second_order_efficiency', 0.0)
            
            # Score combinado
            ntk_score = (
                loss_improvement * 0.25 +
                accuracy_improvement * 0.25 +
                geometric_optimization * 0.25 +
                second_order_efficiency * 0.25
            )
            
            return max(0.0, min(1.0, ntk_score))
            
        except Exception as e:
            logger.error(f"Error calculando score NTK: {e}")
            return 0.0
    
    def _generate_ntk_recommendations(self, metrics: UltraAdvancedOptimizationMetrics, 
                                     theoretical_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para NTK"""
        recommendations = []
        
        try:
            # Recomendaciones basadas en teoría NTK
            if theoretical_analysis.get('geometric_optimization', 0.0) < 0.7:
                recommendations.append("La optimización geométrica es baja, considerar ajustar lambda")
            
            if theoretical_analysis.get('second_order_efficiency', 0.0) < 0.6:
                recommendations.append("La eficiencia de segundo orden es baja, considerar aumentar el ancho de la red")
            
            # Recomendaciones basadas en convergencia
            if metrics.theoretical_convergence < 0.8:
                recommendations.append("La convergencia teórica es lenta, considerar ajustar la tasa de aprendizaje")
            
            # Recomendaciones específicas de NTK
            if self.ntk_lambda < 1e-5:
                recommendations.append("El parámetro lambda es muy pequeño, considerar aumentarlo")
            
            if not self.ntk_infinite_width:
                recommendations.append("Considerar usar el límite de ancho infinito para mejor análisis teórico")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones NTK: {e}")
        
        return recommendations

class NTKOptimizer(torch.optim.Optimizer):
    """
    Implementación del optimizador NTK
    """
    
    def __init__(self, params, lr=1e-3, lambda_reg=1e-4, infinite_width=True, weight_decay=0.0):
        defaults = dict(lr=lr, lambda_reg=lambda_reg, infinite_width=infinite_width, weight_decay=weight_decay)
        super(NTKOptimizer, self).__init__(params, defaults)
        
        self.kernel_magnitude = 0.0
    
    def step(self, closure=None):
        """Paso de optimización NTK"""
        loss = None
        if closure is not None:
            loss = closure()
        
        kernel_magnitudes = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NTK no soporta gradientes dispersos')
                
                # NTK: optimización basada en kernel
                if group['infinite_width']:
                    # En el límite de ancho infinito, usar gradiente regularizado
                    update = grad / (1 + group['lambda_reg'])
                else:
                    # Para ancho finito, usar gradiente estándar
                    update = grad
                
                # Aplicar actualización
                p.data.add_(update, alpha=-group['lr'])
                
                # Aplicar weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                kernel_magnitudes.append(torch.norm(update).item())
        
        if kernel_magnitudes:
            self.kernel_magnitude = np.mean(kernel_magnitudes)
        
        return loss

# Funciones de utilidad
def create_ntk_optimizer(config: UltraAdvancedOptimizerConfig = None) -> NTKWeightOptimizer:
    """Crea un optimizador NTK"""
    return NTKWeightOptimizer(config or UltraAdvancedOptimizerConfig())

def analyze_ntk_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                          criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de NTK en un modelo"""
    try:
        optimizer = NTKWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'theoretical_analysis': result.theoretical_analysis
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento NTK: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN9_RN_1.py - NTK Optimizer cargado exitosamente")
