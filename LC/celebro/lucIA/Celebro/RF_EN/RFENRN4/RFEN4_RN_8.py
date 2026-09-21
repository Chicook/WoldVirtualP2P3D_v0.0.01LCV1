"""
RFEN4_RN_8 - Optimización de Pesos para Ensembles
Implementación de técnicas avanzadas para optimización de pesos en modelos ensemble
Incluye: Ensemble dinámico, optimización colaborativa, y fusión inteligente de pesos
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque
from . import BaseWeightImprover, NeuralWeightConfig, NeuronWeightMetrics, WeightImprovementResult

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuración para optimización de ensembles"""
    ensemble_method: str = "dynamic_ensemble"  # dynamic_ensemble, collaborative_optimization, intelligent_fusion
    ensemble_size: int = 5
    diversity_factor: float = 0.1
    collaboration_strength: float = 0.5
    fusion_method: str = "weighted_average"  # weighted_average, adaptive_fusion, consensus_based
    ensemble_learning_rate: float = 0.01
    diversity_threshold: float = 0.05
    collaboration_frequency: int = 100
    fusion_threshold: float = 0.01
    ensemble_memory_size: int = 1000
    adaptive_weighting: bool = True
    consensus_threshold: float = 0.8


class DynamicEnsembleOptimizer(BaseWeightImprover):
    """
    Optimizador de ensembles dinámicos
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.ensemble_config = EnsembleConfig()
        self.ensemble_models = {}
        self.ensemble_weights = {}
        self.ensemble_performance = {}
        self.diversity_scores = {}
        self.collaboration_history = defaultdict(list)

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando optimización de ensemble dinámico"""

        logger.info("Iniciando optimización de ensemble dinámico")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Crear ensemble de modelos
        self._create_ensemble_models(model)

        # Optimizar ensemble
        neurons_improved = self._optimize_ensemble(model, data_loader)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_ensemble_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=self._calculate_memory_optimization(),
            accuracy_improvement=improvement_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["dynamic_ensemble_optimization"],
            neurons_improved=neurons_improved,
            total_neurons=len(initial_metrics),
            improvement_time=end_time - start_time
        )

    def analyze_neuron_weights(self, model: nn.Module) -> Dict[str, NeuronWeightMetrics]:
        """Analiza los pesos de las neuronas individuales"""

        neuron_metrics = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular métricas para cada parámetro
                metrics = NeuronWeightMetrics(
                    neuron_id=name,
                    weight_magnitude=torch.norm(param.data).item(),
                    weight_variance=torch.var(param.data).item(),
                    gradient_norm=torch.norm(param.grad).item() if param.grad is not None else 0.0,
                    activation_frequency=self._calculate_activation_frequency(param),
                    importance_score=self._calculate_importance_score(param),
                    improvement_rate=0.0,
                    stability_score=self._calculate_stability_score(param),
                    contribution_to_loss=self._estimate_contribution_to_loss(param),
                    timestamp=time.time(),
                    layer_type=self._get_layer_type(name),
                    additional_metrics={
                        'ensemble_weight': self.ensemble_weights.get(name, 1.0),
                        'diversity_score': self.diversity_scores.get(name, 0.0),
                        'ensemble_performance': self.ensemble_performance.get(name, 0.0)
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _create_ensemble_models(self, model: nn.Module) -> None:
        """Crea un ensemble de modelos"""

        self.ensemble_models = {}

        for i in range(self.ensemble_config.ensemble_size):
            # Crear modelo del ensemble
            ensemble_model = {}

            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Crear variación del peso
                    variation = torch.randn_like(param.data) * self.ensemble_config.diversity_factor
                    ensemble_model[name] = param.data + variation

            self.ensemble_models[f'model_{i}'] = ensemble_model

    def _optimize_ensemble(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> int:
        """Optimiza el ensemble de modelos"""

        neurons_improved = 0

        # Calcular pesos del ensemble
        self._calculate_ensemble_weights(model, data_loader)

        # Aplicar optimización del ensemble
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ensemble_weights:
                # Obtener peso del ensemble
                ensemble_weight = self.ensemble_weights[name]

                # Aplicar optimización del ensemble
                if ensemble_weight > self.ensemble_config.fusion_threshold:
                    # Fusionar pesos del ensemble
                    fused_weight = self._fuse_ensemble_weights(name)

                    with torch.no_grad():
                        param.data = fused_weight

                    neurons_improved += 1

        logger.info(f"Optimización de ensemble aplicada a {neurons_improved} neuronas")
        return neurons_improved

    def _calculate_ensemble_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> None:
        """Calcula los pesos del ensemble"""

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular peso del ensemble basado en diversidad y rendimiento
                diversity_score = self._calculate_diversity_score(name)
                performance_score = self._calculate_performance_score(param, name)

                # Peso combinado
                ensemble_weight = diversity_score * 0.5 + performance_score * 0.5

                self.ensemble_weights[name] = ensemble_weight
                self.diversity_scores[name] = diversity_score
                self.ensemble_performance[name] = performance_score

    def _calculate_diversity_score(self, name: str) -> float:
        """Calcula el score de diversidad para un parámetro"""

        if not self.ensemble_models:
            return 0.0

        # Calcular varianza entre modelos del ensemble
        weights = []
        for model_name, model_weights in self.ensemble_models.items():
            if name in model_weights:
                weights.append(model_weights[name])

        if len(weights) > 1:
            # Calcular varianza promedio
            variances = []
            for i in range(len(weights)):
                for j in range(i + 1, len(weights)):
                    variance = torch.var(weights[i] - weights[j]).item()
                    variances.append(variance)

            diversity_score = np.mean(variances) if variances else 0.0
            return min(1.0, diversity_score)

        return 0.0

    def _calculate_performance_score(self, param: torch.Tensor, name: str) -> float:
        """Calcula el score de rendimiento para un parámetro"""

        # Score basado en magnitud y varianza
        magnitude = torch.norm(param.data).item()
        variance = torch.var(param.data).item()

        performance_score = magnitude * 0.6 + variance * 0.4
        return min(1.0, performance_score / 100.0)

    def _fuse_ensemble_weights(self, name: str) -> torch.Tensor:
        """Fusiona los pesos del ensemble"""

        if not self.ensemble_models:
            return torch.zeros(1)

        # Obtener pesos del ensemble
        ensemble_weights = []
        for model_name, model_weights in self.ensemble_models.items():
            if name in model_weights:
                ensemble_weights.append(model_weights[name])

        if not ensemble_weights:
            return torch.zeros(1)

        # Fusionar usando promedio ponderado
        if self.ensemble_config.fusion_method == "weighted_average":
            # Promedio ponderado basado en diversidad
            weights = [self.diversity_scores.get(name, 0.0) for _ in ensemble_weights]
            total_weight = sum(weights)

            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                fused_weight = sum(w * weight for w, weight in zip(ensemble_weights, weights))
            else:
                fused_weight = torch.mean(torch.stack(ensemble_weights), dim=0)

        elif self.ensemble_config.fusion_method == "adaptive_fusion":
            # Fusión adaptativa basada en rendimiento
            performance_weights = [self.ensemble_performance.get(name, 0.0) for _ in ensemble_weights]
            total_performance = sum(performance_weights)

            if total_performance > 0:
                performance_weights = [w / total_performance for w in performance_weights]
                fused_weight = sum(w * weight for w, weight in zip(ensemble_weights, performance_weights))
            else:
                fused_weight = torch.mean(torch.stack(ensemble_weights), dim=0)

        else:  # consensus_based
            # Fusión basada en consenso
            fused_weight = torch.mean(torch.stack(ensemble_weights), dim=0)

        return fused_weight

    def _calculate_activation_frequency(self, param: torch.Tensor) -> float:
        """Calcula la frecuencia de activación"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 5.0)

    def _calculate_importance_score(self, param: torch.Tensor) -> float:
        """Calcula el score de importancia"""

        magnitude = torch.norm(param.data).item()
        variance = torch.var(param.data).item()

        importance = magnitude * (1 + variance)
        return min(1.0, importance / 100.0)

    def _calculate_stability_score(self, param: torch.Tensor) -> float:
        """Calcula el score de estabilidad"""

        return 0.5  # Valor por defecto

    def _estimate_contribution_to_loss(self, param: torch.Tensor) -> float:
        """Estima la contribución a la pérdida"""

        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            return min(1.0, grad_norm / 5.0)
        return 0.0

    def _get_layer_type(self, name: str) -> str:
        """Determina el tipo de capa"""

        if 'conv' in name.lower():
            return 'convolutional'
        elif 'linear' in name.lower() or 'fc' in name.lower():
            return 'linear'
        elif 'bn' in name.lower() or 'norm' in name.lower():
            return 'normalization'
        else:
            return 'other'

    def _calculate_ensemble_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia del ensemble"""

        if not self.ensemble_weights:
            return 0.0

        weights = list(self.ensemble_weights.values())
        convergence_rate = 1.0 / (1.0 + np.std(weights))
        return convergence_rate

    def _calculate_stability_improvement(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """Calcula la mejora en estabilidad"""

        initial_stability = np.mean([m.stability_score for m in initial_metrics.values()])
        final_stability = np.mean([m.stability_score for m in final_metrics.values()])

        return final_stability - initial_stability

    def _calculate_efficiency_gain(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """Calcula la ganancia en eficiencia"""

        initial_efficiency = np.mean([m.importance_score for m in initial_metrics.values()])
        final_efficiency = np.mean([m.importance_score for m in final_metrics.values()])

        return final_efficiency - initial_efficiency

    def _calculate_memory_optimization(self) -> float:
        """Calcula la optimización de memoria"""

        return 0.0  # Implementar según necesidad

    def _calculate_speed_gain(self, optimization_time: float) -> float:
        """Calcula la ganancia en velocidad"""

        expected_time = 8.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain


class CollaborativeOptimizer(BaseWeightImprover):
    """
    Optimizador colaborativo para ensembles
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.ensemble_config = EnsembleConfig(ensemble_method="collaborative_optimization")
        self.collaboration_network = {}
        self.collaboration_weights = {}
        self.collaboration_history = defaultdict(list)
        self.consensus_scores = {}

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando optimización colaborativa"""

        logger.info("Iniciando optimización colaborativa de ensembles")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Crear red de colaboración
        self._create_collaboration_network(model)

        # Aplicar optimización colaborativa
        neurons_improved = self._apply_collaborative_optimization(model, data_loader)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_collaborative_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=0.0,
            accuracy_improvement=improvement_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["collaborative_optimization"],
            neurons_improved=neurons_improved,
            total_neurons=len(initial_metrics),
            improvement_time=end_time - start_time
        )

    def analyze_neuron_weights(self, model: nn.Module) -> Dict[str, NeuronWeightMetrics]:
        """Analiza los pesos de las neuronas individuales"""

        neuron_metrics = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                metrics = NeuronWeightMetrics(
                    neuron_id=name,
                    weight_magnitude=torch.norm(param.data).item(),
                    weight_variance=torch.var(param.data).item(),
                    gradient_norm=torch.norm(param.grad).item() if param.grad is not None else 0.0,
                    activation_frequency=self._calculate_activation_frequency(param),
                    importance_score=self._calculate_importance_score(param),
                    improvement_rate=0.0,
                    stability_score=self._calculate_stability_score(param),
                    contribution_to_loss=self._estimate_contribution_to_loss(param),
                    timestamp=time.time(),
                    layer_type=self._get_layer_type(name),
                    additional_metrics={
                        'collaboration_weight': self.collaboration_weights.get(name, 1.0),
                        'consensus_score': self.consensus_scores.get(name, 0.0),
                        'collaboration_history': self.collaboration_history.get(name, [])[-10:]
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _create_collaboration_network(self, model: nn.Module) -> None:
        """Crea una red de colaboración entre neuronas"""

        self.collaboration_network = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Crear conexiones de colaboración
                collaboration_connections = []

                for other_name, other_param in model.named_parameters():
                    if other_param.requires_grad and other_name != name:
                        # Calcular similitud entre pesos
                        similarity = self._calculate_weight_similarity(param, other_param)

                        if similarity > self.ensemble_config.diversity_threshold:
                            collaboration_connections.append({
                                'name': other_name,
                                'similarity': similarity,
                                'weight': other_param.data.clone()
                            })

                self.collaboration_network[name] = collaboration_connections

    def _calculate_weight_similarity(self, param1: torch.Tensor, param2: torch.Tensor) -> float:
        """Calcula la similitud entre dos pesos"""

        # Calcular similitud usando correlación
        if param1.numel() == param2.numel():
            correlation = torch.corrcoef(torch.stack([param1.flatten(), param2.flatten()]))[0, 1]
            similarity = abs(correlation.item()) if not torch.isnan(correlation) else 0.0
        else:
            similarity = 0.0

        return similarity

    def _apply_collaborative_optimization(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> int:
        """Aplica optimización colaborativa"""

        neurons_improved = 0

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.collaboration_network:
                # Obtener conexiones de colaboración
                collaboration_connections = self.collaboration_network[name]

                if collaboration_connections:
                    # Calcular peso colaborativo
                    collaborative_weight = self._calculate_collaborative_weight(param, collaboration_connections)

                    # Aplicar optimización colaborativa
                    if collaborative_weight is not None:
                        with torch.no_grad():
                            param.data = collaborative_weight

                        # Registrar en historial
                        self._record_collaboration(name, collaborative_weight)

                        neurons_improved += 1

        logger.info(f"Optimización colaborativa aplicada a {neurons_improved} neuronas")
        return neurons_improved

    def _calculate_collaborative_weight(self, param: torch.Tensor, collaboration_connections: List[Dict]) -> Optional[torch.Tensor]:
        """Calcula el peso colaborativo"""

        if not collaboration_connections:
            return None

        # Calcular peso colaborativo usando promedio ponderado
        total_similarity = sum(conn['similarity'] for conn in collaboration_connections)

        if total_similarity > 0:
            collaborative_weight = torch.zeros_like(param.data)

            for conn in collaboration_connections:
                weight = conn['weight']
                similarity = conn['similarity']

                # Normalizar similitud
                normalized_similarity = similarity / total_similarity

                # Añadir contribución
                collaborative_weight += weight * normalized_similarity

            return collaborative_weight

        return None

    def _record_collaboration(self, name: str, collaborative_weight: torch.Tensor) -> None:
        """Registra la colaboración en el historial"""

        self.collaboration_history[name].append({
            'collaborative_weight': collaborative_weight.clone(),
            'timestamp': time.time()
        })

        # Mantener historial limitado
        if len(self.collaboration_history[name]) > 100:
            self.collaboration_history[name].pop(0)

    def _calculate_activation_frequency(self, param: torch.Tensor) -> float:
        """Calcula la frecuencia de activación"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 5.0)

    def _calculate_importance_score(self, param: torch.Tensor) -> float:
        """Calcula el score de importancia"""

        magnitude = torch.norm(param.data).item()
        variance = torch.var(param.data).item()

        importance = magnitude * (1 + variance)
        return min(1.0, importance / 100.0)

    def _calculate_stability_score(self, param: torch.Tensor) -> float:
        """Calcula el score de estabilidad"""

        return 0.5  # Valor por defecto

    def _estimate_contribution_to_loss(self, param: torch.Tensor) -> float:
        """Estima la contribución a la pérdida"""

        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            return min(1.0, grad_norm / 5.0)
        return 0.0

    def _get_layer_type(self, name: str) -> str:
        """Determina el tipo de capa"""

        if 'conv' in name.lower():
            return 'convolutional'
        elif 'linear' in name.lower() or 'fc' in name.lower():
            return 'linear'
        elif 'bn' in name.lower() or 'norm' in name.lower():
            return 'normalization'
        else:
            return 'other'

    def _calculate_collaborative_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia colaborativa"""

        if not self.collaboration_network:
            return 0.0

        convergence_rates = []

        for name, connections in self.collaboration_network.items():
            if len(connections) > 0:
                similarities = [conn['similarity'] for conn in connections]
                convergence_rate = np.mean(similarities)
                convergence_rates.append(convergence_rate)

        return np.mean(convergence_rates) if convergence_rates else 0.0

    def _calculate_stability_improvement(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """Calcula la mejora en estabilidad"""

        initial_stability = np.mean([m.stability_score for m in initial_metrics.values()])
        final_stability = np.mean([m.stability_score for m in final_metrics.values()])

        return final_stability - initial_stability

    def _calculate_efficiency_gain(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """Calcula la ganancia en eficiencia"""

        initial_efficiency = np.mean([m.importance_score for m in initial_metrics.values()])
        final_efficiency = np.mean([m.importance_score for m in final_metrics.values()])

        return final_efficiency - initial_efficiency

    def _calculate_speed_gain(self, optimization_time: float) -> float:
        """Calcula la ganancia en velocidad"""

        expected_time = 10.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain

# Funciones de utilidad


def create_ensemble_optimizer(ensemble_method: str = "dynamic_ensemble") -> BaseWeightImprover:
    """Factory para crear optimizadores de ensemble"""

    config = NeuralWeightConfig()

    if ensemble_method == "dynamic_ensemble":
        return DynamicEnsembleOptimizer(config)
    elif ensemble_method == "collaborative_optimization":
        return CollaborativeOptimizer(config)
    else:
        raise ValueError(f"Método de ensemble no soportado: {ensemble_method}")


def optimize_ensemble_weights(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                              ensemble_method: str = "dynamic_ensemble") -> WeightImprovementResult:
    """Función de conveniencia para optimizar pesos de ensemble"""

    ensemble_optimizer = create_ensemble_optimizer(ensemble_method)
    return ensemble_optimizer.improve_weights(model, data_loader)


# Exportar clases y funciones principales
__all__ = [
    'EnsembleConfig',
    'DynamicEnsembleOptimizer',
    'CollaborativeOptimizer',
    'create_ensemble_optimizer',
    'optimize_ensemble_weights'
]

logger.info("RFEN4_RN_8 - Optimización de Pesos para Ensembles cargada correctamente")
