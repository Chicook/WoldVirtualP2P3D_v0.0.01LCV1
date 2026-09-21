"""
RFEN8_RN_1.py - SAM (Sharpness Aware Minimization) Optimizer
============================================================

Implementación del optimizador SAM (Sharpness Aware Minimization) para
mejorar la generalización de redes neuronales mediante la minimización
de la sharpness de la función de pérdida.

Características principales:
- Minimización de sharpness para mejor generalización
- Optimización adaptativa del radio de perturbación
- Soporte para diferentes estrategias de perturbación
- Integración con optimizadores base (SGD, Adam, etc.)
- Análisis de convergencia y estabilidad

Referencias:
- Foret, P., et al. "Sharpness-Aware Minimization for Efficiently Improving Generalization"
- Implementación basada en PyTorch SAM
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

class SAMWeightOptimizer(BaseAdvancedOptimizer):
    """
    Optimizador SAM (Sharpness Aware Minimization) para mejorar la generalización
    mediante la minimización de la sharpness de la función de pérdida.
    """
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.base_optimizer = None
        self.sam_rho = config.sam_rho
        self.sam_adaptive = config.sam_adaptive
        self.perturbation_history = []
        self.sharpness_history = []
        self.generalization_metrics = {}
        self.perturbation_strategy = "l2"  # l2, linf, adaptive
        self.epsilon_adaptive = True
        self.epsilon_decay = 0.99
        self.min_epsilon = 1e-6
        self.current_epsilon = self.sam_rho
        
        logger.info(f"SAMWeightOptimizer inicializado con rho={self.sam_rho}, adaptive={self.sam_adaptive}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador SAM con optimizador base"""
        try:
            # Crear optimizador base (Adam por defecto)
            self.base_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon
            )
            
            # Crear SAM wrapper
            sam_optimizer = SAM(
                self.base_optimizer,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive,
                perturbation_strategy=self.perturbation_strategy
            )
            
            self.optimizer = sam_optimizer
            logger.info("Optimizador SAM creado exitosamente")
            return sam_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador SAM: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos del modelo usando SAM"""
        try:
            logger.info("Iniciando optimización SAM")
            start_time = time.time()
            
            # Configurar criterio de pérdida
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            # Crear optimizador SAM
            optimizer = self.create_optimizer(model)
            
            # Métricas iniciales
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            initial_loss = initial_metrics['loss']
            initial_accuracy = initial_metrics['accuracy']
            
            # Entrenamiento SAM
            model.train()
            loss_history = []
            accuracy_history = []
            sharpness_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    # Paso SAM
                    loss = self._sam_step(model, data, target, optimizer, criterion)
                    
                    epoch_losses.append(loss.item())
                    
                    # Calcular precisión
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    # Actualizar métricas de sharpness
                    if batch_idx % 10 == 0:
                        sharpness = self._calculate_sharpness(model, data, target, criterion)
                        sharpness_history.append(sharpness)
                
                # Métricas de época
                avg_loss = np.mean(epoch_losses)
                avg_accuracy = np.mean(epoch_accuracies)
                loss_history.append(avg_loss)
                accuracy_history.append(avg_accuracy)
                
                # Actualizar epsilon adaptativo
                if self.epsilon_adaptive:
                    self._update_adaptive_epsilon(avg_loss, loss_history)
                
                # Verificar convergencia
                if self._check_convergence(loss_history):
                    logger.info(f"Convergencia alcanzada en época {epoch}")
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
            convergence_speed = 1.0 / max(convergence_iterations, 1)
            generalization_improvement = final_accuracy - initial_accuracy
            loss_reduction = (initial_loss - final_loss) / max(initial_loss, 1e-8)
            
            # Análisis de sharpness
            sharpness_analysis = self._analyze_sharpness(sharpness_history)
            
            # Crear métricas
            metrics = OptimizationMetrics(
                optimizer_name="SAM",
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
                optimization_robustness=sharpness_analysis['robustness'],
                overall_score=self._calculate_sam_score(initial_metrics, final_metrics, sharpness_analysis),
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
                    'sharpness_analysis': sharpness_analysis,
                    'generalization_metrics': self.generalization_metrics,
                    'perturbation_history': self.perturbation_history
                },
                recommendations=self._generate_sam_recommendations(metrics, sharpness_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización SAM completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización SAM: {e}")
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
    
    def _sam_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                  optimizer: 'SAM', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización SAM"""
        try:
            # Primer paso: calcular gradientes en el punto actual
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss
            
            # Paso SAM
            loss = optimizer.step(closure)
            
            # Registrar perturbación
            if hasattr(optimizer, 'perturbation_magnitude'):
                self.perturbation_history.append(optimizer.perturbation_magnitude)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error en paso SAM: {e}")
            raise
    
    def _calculate_sharpness(self, model: nn.Module, data: torch.Tensor, 
                           target: torch.Tensor, criterion: nn.Module) -> float:
        """Calcula la sharpness de la función de pérdida"""
        try:
            # Obtener pesos actuales
            current_weights = {name: param.clone() for name, param in model.named_parameters()}
            
            # Calcular pérdida en el punto actual
            with torch.no_grad():
                output = model(data)
                current_loss = criterion(output, target).item()
            
            # Perturbar pesos y calcular pérdida
            max_perturbation_loss = current_loss
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Perturbación pequeña
                    perturbation = torch.randn_like(param) * self.current_epsilon
                    param.data += perturbation
                    
                    with torch.no_grad():
                        output = model(data)
                        perturbed_loss = criterion(output, target).item()
                        max_perturbation_loss = max(max_perturbation_loss, perturbed_loss)
                    
                    # Restaurar peso original
                    param.data = current_weights[name]
            
            # Calcular sharpness
            sharpness = (max_perturbation_loss - current_loss) / max(current_loss, 1e-8)
            return sharpness
            
        except Exception as e:
            logger.error(f"Error calculando sharpness: {e}")
            return 0.0
    
    def _update_adaptive_epsilon(self, current_loss: float, loss_history: List[float]) -> None:
        """Actualiza epsilon adaptativamente basado en el progreso"""
        try:
            if len(loss_history) < 5:
                return
            
            # Calcular tendencia de pérdida
            recent_losses = loss_history[-5:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            # Ajustar epsilon basado en la tendencia
            if loss_trend < -0.01:  # Mejorando rápidamente
                self.current_epsilon *= 1.01
            elif loss_trend > 0.01:  # Empeorando
                self.current_epsilon *= 0.99
            
            # Mantener límites
            self.current_epsilon = max(self.min_epsilon, 
                                     min(self.current_epsilon, self.sam_rho * 2))
            
        except Exception as e:
            logger.error(f"Error actualizando epsilon adaptativo: {e}")
    
    def _analyze_sharpness(self, sharpness_history: List[float]) -> Dict:
        """Analiza la evolución de la sharpness"""
        try:
            if not sharpness_history:
                return {'robustness': 0.0, 'stability': 0.0, 'trend': 'stable'}
            
            # Calcular métricas de sharpness
            mean_sharpness = np.mean(sharpness_history)
            std_sharpness = np.std(sharpness_history)
            
            # Calcular tendencia
            if len(sharpness_history) > 1:
                trend = np.polyfit(range(len(sharpness_history)), sharpness_history, 1)[0]
                if trend < -0.001:
                    trend_str = 'decreasing'
                elif trend > 0.001:
                    trend_str = 'increasing'
                else:
                    trend_str = 'stable'
            else:
                trend_str = 'stable'
            
            # Calcular robustez (menor sharpness = mayor robustez)
            robustness = max(0.0, 1.0 - mean_sharpness)
            stability = max(0.0, 1.0 - std_sharpness)
            
            return {
                'mean_sharpness': mean_sharpness,
                'std_sharpness': std_sharpness,
                'robustness': robustness,
                'stability': stability,
                'trend': trend_str,
                'final_sharpness': sharpness_history[-1] if sharpness_history else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analizando sharpness: {e}")
            return {'robustness': 0.0, 'stability': 0.0, 'trend': 'stable'}
    
    def _calculate_sam_score(self, initial_metrics: Dict, final_metrics: Dict, 
                           sharpness_analysis: Dict) -> float:
        """Calcula el score específico de SAM"""
        try:
            # Mejora de pérdida
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            
            # Mejora de precisión
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            
            # Score de robustez (sharpness)
            robustness_score = sharpness_analysis.get('robustness', 0.0)
            
            # Score de estabilidad
            stability_score = sharpness_analysis.get('stability', 0.0)
            
            # Score combinado
            sam_score = (
                loss_improvement * 0.3 +
                accuracy_improvement * 0.3 +
                robustness_score * 0.25 +
                stability_score * 0.15
            )
            
            return max(0.0, min(1.0, sam_score))
            
        except Exception as e:
            logger.error(f"Error calculando score SAM: {e}")
            return 0.0
    
    def _generate_sam_recommendations(self, metrics: OptimizationMetrics, 
                                    sharpness_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para SAM"""
        recommendations = []
        
        try:
            # Recomendaciones basadas en sharpness
            if sharpness_analysis.get('robustness', 0.0) < 0.5:
                recommendations.append("Considerar aumentar el radio de perturbación SAM para mejorar la robustez")
            
            if sharpness_analysis.get('stability', 0.0) < 0.7:
                recommendations.append("La estabilidad de sharpness es baja, considerar ajustar el epsilon adaptativo")
            
            # Recomendaciones basadas en convergencia
            if metrics.convergence_speed < 0.01:
                recommendations.append("La convergencia es lenta, considerar aumentar la tasa de aprendizaje")
            
            # Recomendaciones basadas en generalización
            if metrics.generalization_improvement < 0.05:
                recommendations.append("La mejora de generalización es limitada, considerar técnicas adicionales de regularización")
            
            # Recomendaciones específicas de SAM
            if self.current_epsilon < self.min_epsilon * 2:
                recommendations.append("El epsilon adaptativo es muy pequeño, considerar reinicializar")
            
            if len(self.perturbation_history) > 0:
                avg_perturbation = np.mean(self.perturbation_history)
                if avg_perturbation < self.sam_rho * 0.5:
                    recommendations.append("Las perturbaciones son pequeñas, considerar aumentar rho")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones SAM: {e}")
        
        return recommendations

class SAM(torch.optim.Optimizer):
    """
    Implementación del optimizador SAM (Sharpness Aware Minimization)
    """
    
    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05,
                 adaptive: bool = True, perturbation_strategy: str = "l2"):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.perturbation_strategy = perturbation_strategy
        self.perturbation_magnitude = 0.0
        
        # Copiar parámetros del optimizador base
        defaults = dict(rho=rho, adaptive=adaptive, perturbation_strategy=perturbation_strategy)
        super(SAM, self).__init__(base_optimizer.param_groups, defaults)
    
    def step(self, closure=None):
        """Paso de optimización SAM"""
        if closure is None:
            raise ValueError("SAM requiere una función closure")
        
        # Primer paso: calcular gradientes
        loss = closure()
        
        # Calcular perturbación
        perturbation = self._calculate_perturbation()
        
        # Aplicar perturbación
        self._apply_perturbation(perturbation)
        
        # Segundo paso: calcular gradientes en el punto perturbado
        loss = closure()
        
        # Paso del optimizador base
        self.base_optimizer.step()
        
        # Remover perturbación
        self._remove_perturbation(perturbation)
        
        return loss
    
    def _calculate_perturbation(self) -> List[torch.Tensor]:
        """Calcula la perturbación SAM"""
        perturbation = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Calcular norma del gradiente
                grad_norm = torch.norm(p.grad, p=2)
                
                if grad_norm > 0:
                    # Calcular perturbación
                    if self.perturbation_strategy == "l2":
                        eps = self.rho / (grad_norm + 1e-8)
                        pert = eps * p.grad
                    elif self.perturbation_strategy == "linf":
                        eps = self.rho
                        pert = eps * torch.sign(p.grad)
                    else:  # adaptive
                        eps = self.rho / (grad_norm + 1e-8)
                        pert = eps * p.grad
                    
                    perturbation.append(pert)
                    self.perturbation_magnitude = torch.norm(pert, p=2).item()
                else:
                    perturbation.append(torch.zeros_like(p))
        
        return perturbation
    
    def _apply_perturbation(self, perturbation: List[torch.Tensor]) -> None:
        """Aplica la perturbación a los parámetros"""
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.add_(perturbation[idx])
                    idx += 1
    
    def _remove_perturbation(self, perturbation: List[torch.Tensor]) -> None:
        """Remueve la perturbación de los parámetros"""
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.sub_(perturbation[idx])
                    idx += 1

# Funciones de utilidad
def create_sam_optimizer(config: AdvancedOptimizerConfig = None) -> SAMWeightOptimizer:
    """Crea un optimizador SAM"""
    return SAMWeightOptimizer(config or AdvancedOptimizerConfig())

def analyze_sam_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                          criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de SAM en un modelo"""
    try:
        optimizer = SAMWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'sharpness_analysis': result.performance_analysis.get('sharpness_analysis', {})
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento SAM: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_1.py - SAM Optimizer cargado exitosamente")
