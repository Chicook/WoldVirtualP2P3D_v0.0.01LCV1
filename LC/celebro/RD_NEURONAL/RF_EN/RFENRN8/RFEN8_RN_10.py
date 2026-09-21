"""
RFEN8_RN_10.py - Sistema Integrado de Optimizadores Avanzados 2025+
===================================================================

Sistema integrado que combina todos los optimizadores avanzados implementados
en RFENRN8 para proporcionar la mejor optimización de pesos neuronales.

Características principales:
- Integración inteligente de múltiples optimizadores
- Selección adaptativa basada en rendimiento
- Combinación híbrida de técnicas
- Análisis comparativo automático
- Recomendaciones inteligentes

Optimizadores integrados:
- SAM (Sharpness Aware Minimization)
- Lion (Google 2023)
- AdaBelief
- RAdam (Rectified Adam)
- AdaBound
- Lookahead con Ranger
- NovoGrad
- SWATS (Switching from Adam to SGD)
- QHAdam (Quasi-Hyperbolic Adam)
"""

try:
    import torch
    import torch.nn as nn
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

class IntegratedAdvancedOptimizer(BaseAdvancedOptimizer):
    """Sistema integrado de optimizadores avanzados 2025+"""
    
    def __init__(self, config: AdvancedOptimizerConfig):
        super().__init__(config)
        self.optimizers = {}
        self.optimizer_weights = {}
        self.performance_history = defaultdict(list)
        self.adaptive_selection = True
        self.hybrid_mode = True
        self.comparison_metrics = {}
        
        # Inicializar optimizadores
        self._initialize_optimizers()
        
        logger.info(f"IntegratedAdvancedOptimizer inicializado con {len(self.optimizers)} optimizadores")
    
    def _initialize_optimizers(self):
        """Inicializa todos los optimizadores avanzados"""
        try:
            # Importar optimizadores
            from .RFEN8_RN_1 import create_sam_optimizer
            from .RFEN8_RN_2 import create_lion_optimizer
            from .RFEN8_RN_3 import create_adabelief_optimizer
            from .RFEN8_RN_4 import create_radam_optimizer
            from .RFEN8_RN_5 import create_adabound_optimizer
            from .RFEN8_RN_6 import create_lookahead_optimizer
            from .RFEN8_RN_7 import create_novograd_optimizer
            from .RFEN8_RN_8 import create_swats_optimizer
            from .RFEN8_RN_9 import create_qhadam_optimizer
            
            # Crear optimizadores
            self.optimizers = {
                'sam': create_sam_optimizer(self.config),
                'lion': create_lion_optimizer(self.config),
                'adabelief': create_adabelief_optimizer(self.config),
                'radam': create_radam_optimizer(self.config),
                'adabound': create_adabound_optimizer(self.config),
                'lookahead': create_lookahead_optimizer(self.config),
                'novograd': create_novograd_optimizer(self.config),
                'swats': create_swats_optimizer(self.config),
                'qhadam': create_qhadam_optimizer(self.config)
            }
            
            # Pesos iniciales iguales
            initial_weight = 1.0 / len(self.optimizers)
            self.optimizer_weights = {name: initial_weight for name in self.optimizers.keys()}
            
            logger.info("Todos los optimizadores avanzados inicializados exitosamente")
            
        except Exception as e:
            logger.error(f"Error inicializando optimizadores: {e}")
            raise
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Crea el optimizador integrado"""
        try:
            # Por ahora, usar el optimizador con mejor peso
            best_optimizer_name = max(self.optimizer_weights, key=self.optimizer_weights.get)
            best_optimizer = self.optimizers[best_optimizer_name]
            
            return best_optimizer.create_optimizer(model)
            
        except Exception as e:
            logger.error(f"Error creando optimizador integrado: {e}")
            raise
    
    def optimize_weights(self, model: nn.Module, 
                        data_loader: torch.utils.data.DataLoader,
                        criterion: nn.Module = None) -> OptimizationResult:
        """Optimiza los pesos usando el sistema integrado"""
        try:
            logger.info("Iniciando optimización integrada avanzada")
            start_time = time.time()
            
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            
            # Comparar todos los optimizadores
            comparison_results = self._compare_all_optimizers(model, data_loader, criterion)
            
            # Seleccionar mejor optimizador
            best_optimizer_name = self._select_best_optimizer(comparison_results)
            best_optimizer = self.optimizers[best_optimizer_name]
            
            logger.info(f"Optimizador seleccionado: {best_optimizer_name}")
            
            # Ejecutar optimización con el mejor optimizador
            result = best_optimizer.optimize_weights(model, data_loader, criterion)
            
            # Actualizar pesos de optimizadores
            self._update_optimizer_weights(comparison_results)
            
            # Crear resultado integrado
            integrated_result = self._create_integrated_result(result, comparison_results, best_optimizer_name)
            
            optimization_time = time.time() - start_time
            integrated_result.metrics.optimization_time = optimization_time
            
            logger.info(f"Optimización integrada completada exitosamente. Score: {integrated_result.metrics.overall_score:.4f}")
            return integrated_result
            
        except Exception as e:
            logger.error(f"Error en optimización integrada: {e}")
            return OptimizationResult(
                success=False, optimized_model=None, metrics=None,
                optimization_history=[], best_weights=None,
                convergence_analysis={}, performance_analysis={},
                recommendations=[], error_message=str(e)
            )
    
    def _compare_all_optimizers(self, model: nn.Module, 
                               data_loader: torch.utils.data.DataLoader,
                               criterion: nn.Module) -> Dict:
        """Compara el rendimiento de todos los optimizadores"""
        comparison_results = {}
        
        for name, optimizer in self.optimizers.items():
            try:
                logger.info(f"Comparando optimizador: {name}")
                
                # Crear copia del modelo para cada optimizador
                model_copy = copy.deepcopy(model)
                
                # Ejecutar optimización
                result = optimizer.optimize_weights(model_copy, data_loader, criterion)
                
                if result.success:
                    comparison_results[name] = {
                        'score': result.metrics.overall_score,
                        'convergence_speed': result.metrics.convergence_speed,
                        'generalization_improvement': result.metrics.generalization_improvement,
                        'training_stability': result.metrics.training_stability,
                        'optimization_robustness': result.metrics.optimization_robustness,
                        'result': result
                    }
                else:
                    comparison_results[name] = {
                        'score': 0.0,
                        'convergence_speed': 0.0,
                        'generalization_improvement': 0.0,
                        'training_stability': 0.0,
                        'optimization_robustness': 0.0,
                        'result': None
                    }
                
            except Exception as e:
                logger.error(f"Error comparando optimizador {name}: {e}")
                comparison_results[name] = {
                    'score': 0.0,
                    'convergence_speed': 0.0,
                    'generalization_improvement': 0.0,
                    'training_stability': 0.0,
                    'optimization_robustness': 0.0,
                    'result': None
                }
        
        return comparison_results
    
    def _select_best_optimizer(self, comparison_results: Dict) -> str:
        """Selecciona el mejor optimizador basado en comparación"""
        try:
            # Calcular score combinado para cada optimizador
            optimizer_scores = {}
            
            for name, results in comparison_results.items():
                if results['result'] is not None:
                    # Score combinado con pesos
                    combined_score = (
                        results['score'] * 0.3 +
                        results['convergence_speed'] * 0.2 +
                        results['generalization_improvement'] * 0.2 +
                        results['training_stability'] * 0.15 +
                        results['optimization_robustness'] * 0.15
                    )
                    optimizer_scores[name] = combined_score
                else:
                    optimizer_scores[name] = 0.0
            
            # Seleccionar el mejor
            best_optimizer = max(optimizer_scores, key=optimizer_scores.get)
            
            logger.info(f"Mejor optimizador seleccionado: {best_optimizer} (score: {optimizer_scores[best_optimizer]:.4f})")
            return best_optimizer
            
        except Exception as e:
            logger.error(f"Error seleccionando mejor optimizador: {e}")
            return 'adam'  # Fallback
    
    def _update_optimizer_weights(self, comparison_results: Dict):
        """Actualiza los pesos de los optimizadores basado en rendimiento"""
        try:
            if not self.adaptive_selection:
                return
            
            # Calcular nuevos pesos basados en rendimiento
            total_score = sum(results['score'] for results in comparison_results.values())
            
            if total_score > 0:
                for name, results in comparison_results.items():
                    new_weight = results['score'] / total_score
                    # Suavizar la actualización
                    self.optimizer_weights[name] = 0.7 * self.optimizer_weights[name] + 0.3 * new_weight
                
                # Normalizar pesos
                total_weight = sum(self.optimizer_weights.values())
                for name in self.optimizer_weights:
                    self.optimizer_weights[name] /= total_weight
            
        except Exception as e:
            logger.error(f"Error actualizando pesos de optimizadores: {e}")
    
    def _create_integrated_result(self, best_result: OptimizationResult, 
                                 comparison_results: Dict, 
                                 best_optimizer_name: str) -> OptimizationResult:
        """Crea el resultado integrado"""
        try:
            # Crear métricas integradas
            integrated_metrics = OptimizationMetrics(
                optimizer_name=f"IntegratedAdvanced_{best_optimizer_name}",
                initial_loss=best_result.metrics.initial_loss,
                final_loss=best_result.metrics.final_loss,
                convergence_iterations=best_result.metrics.convergence_iterations,
                convergence_speed=best_result.metrics.convergence_speed,
                generalization_improvement=best_result.metrics.generalization_improvement,
                computational_efficiency=best_result.metrics.computational_efficiency,
                memory_usage=best_result.metrics.memory_usage,
                training_stability=best_result.metrics.training_stability,
                test_accuracy=best_result.metrics.test_accuracy,
                loss_reduction=best_result.metrics.loss_reduction,
                gradient_norm=best_result.metrics.gradient_norm,
                weight_magnitude=best_result.metrics.weight_magnitude,
                optimization_robustness=best_result.metrics.optimization_robustness,
                overall_score=best_result.metrics.overall_score,
                optimization_time=best_result.metrics.optimization_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Crear análisis de comparación
            comparison_analysis = {
                'best_optimizer': best_optimizer_name,
                'optimizer_scores': {name: results['score'] for name, results in comparison_results.items()},
                'optimizer_weights': self.optimizer_weights.copy(),
                'comparison_summary': self._create_comparison_summary(comparison_results)
            }
            
            # Crear recomendaciones integradas
            integrated_recommendations = self._generate_integrated_recommendations(
                best_result, comparison_results, best_optimizer_name
            )
            
            return OptimizationResult(
                success=best_result.success,
                optimized_model=best_result.optimized_model,
                metrics=integrated_metrics,
                optimization_history=best_result.optimization_history,
                best_weights=best_result.best_weights,
                convergence_analysis=best_result.convergence_analysis,
                performance_analysis={
                    'comparison_analysis': comparison_analysis,
                    'best_optimizer_analysis': best_result.performance_analysis
                },
                recommendations=integrated_recommendations,
                error_message=best_result.error_message
            )
            
        except Exception as e:
            logger.error(f"Error creando resultado integrado: {e}")
            return best_result
    
    def _create_comparison_summary(self, comparison_results: Dict) -> Dict:
        """Crea un resumen de la comparación"""
        try:
            summary = {
                'total_optimizers': len(comparison_results),
                'successful_optimizers': sum(1 for results in comparison_results.values() if results['result'] is not None),
                'average_score': np.mean([results['score'] for results in comparison_results.values()]),
                'score_std': np.std([results['score'] for results in comparison_results.values()]),
                'best_score': max(results['score'] for results in comparison_results.values()),
                'worst_score': min(results['score'] for results in comparison_results.values())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creando resumen de comparación: {e}")
            return {}
    
    def _generate_integrated_recommendations(self, best_result: OptimizationResult,
                                           comparison_results: Dict,
                                           best_optimizer_name: str) -> List[str]:
        """Genera recomendaciones integradas"""
        recommendations = []
        
        try:
            # Recomendaciones del mejor optimizador
            recommendations.extend(best_result.recommendations)
            
            # Recomendaciones basadas en comparación
            if len(comparison_results) > 1:
                scores = [results['score'] for results in comparison_results.values()]
                if np.std(scores) < 0.1:
                    recommendations.append("Los optimizadores tienen rendimiento similar, considerar usar el más estable")
                
                # Identificar optimizadores con bajo rendimiento
                low_performers = [name for name, results in comparison_results.items() 
                                if results['score'] < 0.5]
                if low_performers:
                    recommendations.append(f"Optimizadores con bajo rendimiento: {', '.join(low_performers)}")
            
            # Recomendaciones específicas del sistema integrado
            if self.optimizer_weights[best_optimizer_name] > 0.5:
                recommendations.append(f"{best_optimizer_name} es consistentemente el mejor optimizador")
            
            recommendations.append("Considerar usar el sistema integrado para optimización automática")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones integradas: {e}")
        
        return recommendations

def create_integrated_advanced_optimizer(config: AdvancedOptimizerConfig = None) -> IntegratedAdvancedOptimizer:
    """Crea el sistema integrado de optimizadores avanzados"""
    return IntegratedAdvancedOptimizer(config or AdvancedOptimizerConfig())

def analyze_integrated_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                  criterion: nn.Module = None) -> Dict:
    """Analiza el rendimiento del sistema integrado"""
    try:
        optimizer = IntegratedAdvancedOptimizer()
        result = optimizer.optimize_weights(model, data_loader, criterion)
        
        return {
            'success': result.success,
            'metrics': result.metrics,
            'recommendations': result.recommendations,
            'comparison_analysis': result.performance_analysis.get('comparison_analysis', {}),
            'best_optimizer': result.performance_analysis.get('comparison_analysis', {}).get('best_optimizer', 'unknown')
        }
    except Exception as e:
        logger.error(f"Error analizando rendimiento integrado: {e}")
        return {'success': False, 'error': str(e)}

logger.info("RFEN8_RN_10.py - Sistema Integrado de Optimizadores Avanzados cargado exitosamente")
