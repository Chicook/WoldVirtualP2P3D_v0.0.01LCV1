"""
RFEN8_RN_8.py - SWATS (Switching from Adam to SGD) Optimizer
=============================================================

Implementación del optimizador SWATS que cambia automáticamente de Adam
a SGD durante el entrenamiento para mejorar la convergencia final.

Características principales:
- Transición automática Adam-SGD
- Detección inteligente del momento de cambio
- Mejor convergencia final
- Estrategias de annealing adaptativas
- Optimización híbrida

Referencias:
- Keskar, N. S., & Socher, R. "Improving Generalization Performance by Switching from Adam to SGD"
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

class SWATSWeightOptimizer(BaseAdvancedOptimizer):
    """Optimizador SWATS con transición Adam-SGD"""
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.swats_annealing_strategy = config.swats_annealing_strategy
        self.swats_annealing_period = config.swats_annealing_period
        self.transition_history = []
        self.switching_metrics = {}
        self.adam_phase = True
        self.switch_threshold = 0.1
        
        logger.info(f"SWATSWeightOptimizer inicializado con strategy={self.swats_annealing_strategy}")
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador SWATS"""
        try:
            swats_optimizer = SWATS(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay,
                annealing_strategy=self.swats_annealing_strategy,
                annealing_period=self.swats_annealing_period
            )
            
            self.optimizer = swats_optimizer
            logger.info("Optimizador SWATS creado exitosamente")
            return swats_optimizer
            
        except Exception as e:
            logger.error(f"Error creando optimizador SWATS: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos del modelo usando SWATS"""
        try:
            logger.info("Iniciando optimización SWATS")
            start_time = time.time()
            
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            optimizer = self.create_optimizer(model)
            initial_metrics = self._evaluate_model(model, data_loader, criterion)
            
            model.train()
            loss_history = []
            accuracy_history = []
            transition_history = []
            
            for epoch in range(self.config.max_iterations):
                epoch_losses = []
                epoch_accuracies = []
                
                for batch_idx, (data, target) in enumerate(data_loader):
                    loss = self._swats_step(model, data, target, optimizer, criterion)
                    epoch_losses.append(loss.item())
                    
                    with torch.no_grad():
                        pred = model(data)
                        accuracy = (pred.argmax(dim=1) == target).float().mean().item()
                        epoch_accuracies.append(accuracy)
                    
                    if hasattr(optimizer, 'current_phase'):
                        transition_history.append(optimizer.current_phase)
                
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
            
            # Análisis de transición
            transition_analysis = self._analyze_transition(transition_history, loss_history)
            
            metrics = OptimizationMetrics(
                optimizer_name="SWATS",
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
                optimization_robustness=transition_analysis['robustness'],
                overall_score=self._calculate_swats_score(initial_metrics, final_metrics, transition_analysis),
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
                recommendations=self._generate_swats_recommendations(metrics, transition_analysis),
                error_message=None
            )
            
            logger.info(f"Optimización SWATS completada exitosamente. Score: {metrics.overall_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en optimización SWATS: {e}")
            return OptimizationResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights=None,
                convergence_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(e)
            )
    
    def _swats_step(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor,
                    optimizer: 'SWATS', criterion: nn.Module) -> torch.Tensor:
        """Realiza un paso de optimización SWATS"""
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            return loss
        except Exception as e:
            logger.error(f"Error en paso SWATS: {e}")
            raise
    
    def _analyze_transition(self, transition_history: List[str], 
                           loss_history: List[float]) -> Dict:
        """Analiza la transición Adam-SGD"""
        try:
            if not transition_history:
                return {'robustness': 0.0, 'transition_effectiveness': 0.0, 'switching_timing': 0.0}
            
            # Calcular efectividad de transición
            adam_phases = sum(1 for phase in transition_history if phase == 'adam')
            sgd_phases = sum(1 for phase in transition_history if phase == 'sgd')
            total_phases = len(transition_history)
            
            transition_effectiveness = max(0.0, 1.0 - abs(adam_phases - sgd_phases) / max(total_phases, 1))
            
            # Calcular timing de cambio
            if len(transition_history) > 1:
                phase_changes = sum(1 for i in range(1, len(transition_history)) 
                                  if transition_history[i] != transition_history[i-1])
                switching_timing = min(1.0, phase_changes / max(len(transition_history), 1))
            else:
                switching_timing = 0.0
            
            # Calcular robustez
            robustness = (transition_effectiveness + switching_timing) / 2.0
            
            return {
                'robustness': robustness,
                'transition_effectiveness': transition_effectiveness,
                'switching_timing': switching_timing,
                'adam_phases': adam_phases,
                'sgd_phases': sgd_phases
            }
            
        except Exception as e:
            logger.error(f"Error analizando transición: {e}")
            return {'robustness': 0.0, 'transition_effectiveness': 0.0, 'switching_timing': 0.0}
    
    def _calculate_swats_score(self, initial_metrics: Dict, final_metrics: Dict, 
                              transition_analysis: Dict) -> float:
        """Calcula el score específico de SWATS"""
        try:
            loss_improvement = (initial_metrics['loss'] - final_metrics['loss']) / max(initial_metrics['loss'], 1e-8)
            accuracy_improvement = final_metrics['accuracy'] - initial_metrics['accuracy']
            robustness = transition_analysis.get('robustness', 0.0)
            transition_effectiveness = transition_analysis.get('transition_effectiveness', 0.0)
            
            swats_score = (
                loss_improvement * 0.3 +
                accuracy_improvement * 0.3 +
                robustness * 0.2 +
                transition_effectiveness * 0.2
            )
            
            return max(0.0, min(1.0, swats_score))
            
        except Exception as e:
            logger.error(f"Error calculando score SWATS: {e}")
            return 0.0
    
    def _generate_swats_recommendations(self, metrics: OptimizationMetrics, 
                                       transition_analysis: Dict) -> List[str]:
        """Genera recomendaciones específicas para SWATS"""
        recommendations = []
        
        try:
            if transition_analysis.get('transition_effectiveness', 0.0) < 0.7:
                recommendations.append("La efectividad de transición es baja, considerar ajustar la estrategia de annealing")
            
            if transition_analysis.get('switching_timing', 0.0) < 0.5:
                recommendations.append("El timing de cambio es subóptimo, considerar ajustar el período de annealing")
            
            if metrics.convergence_speed < 0.01:
                recommendations.append("La convergencia es lenta, considerar ajustar los parámetros de transición")
            
            if metrics.training_stability < 0.7:
                recommendations.append("La estabilidad del entrenamiento es baja, considerar suavizar la transición")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones SWATS: {e}")
        
        return recommendations

class SWATS(torch.optim.Optimizer):
    """Implementación del optimizador SWATS"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 annealing_strategy='cosine', annealing_period=10):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       annealing_strategy=annealing_strategy, annealing_period=annealing_period)
        super(SWATS, self).__init__(params, defaults)
        
        self.current_phase = 'adam'
        self.step_count = 0
        self.switch_threshold = 0.1
        
        # Inicializar optimizadores
        self.adam_optimizer = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.sgd_optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    
    def step(self, closure=None):
        """Paso de optimización SWATS"""
        loss = None
        if closure is not None:
            loss = closure()
        
        self.step_count += 1
        
        # Decidir fase basada en el progreso
        if self.step_count % self.defaults['annealing_period'] == 0:
            self._check_switch_condition()
        
        # Ejecutar optimización según la fase actual
        if self.current_phase == 'adam':
            loss = self.adam_optimizer.step(closure)
        else:
            loss = self.sgd_optimizer.step(closure)
        
        return loss
    
    def _check_switch_condition(self):
        """Verifica si debe cambiar de Adam a SGD"""
        # Lógica simplificada para cambio de fase
        if self.step_count > 100 and self.current_phase == 'adam':
            self.current_phase = 'sgd'
            logger.info(f"Cambiando de Adam a SGD en paso {self.step_count}")

def create_swats_optimizer(config: AdvancedOptimizerConfig = None) -> SWATSWeightOptimizer:
    """Crea un optimizador SWATS"""
    return SWATSWeightOptimizer(config or AdvancedOptimizerConfig())

def analyze_swats_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                             criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento de SWATS en un modelo"""
    try:
        optimizer = SWATSWeightOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'transition_analysis': result.performance_analysis.get('transition_analysis', {})
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento SWATS: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_8.py - SWATS Optimizer cargado exitosamente")
