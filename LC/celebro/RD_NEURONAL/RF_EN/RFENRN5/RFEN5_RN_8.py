try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    pass  # dependencia pesada opcional
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import random
import copy
from collections import defaultdict

# Configuración del logger
logger = logging.getLogger(__name__)


class AdaptiveEnsembleOptimizer(ABC):
    """
    Clase base abstracta para sistemas de ensembles adaptativos para optimización de pesos.
    Define la interfaz común para todas las estrategias de ensemble adaptativo.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else {}
        logger.info("AdaptiveEnsembleOptimizer base inicializado.")

    @abstractmethod
    def optimize_ensemble_weights(self, ensemble_models: List[nn.Module], data_loader=None) -> List[nn.Module]:
        """
        Método abstracto para optimizar los pesos de un ensemble de modelos.
        Debe ser implementado por las subclases.
        """
        pass

    def _evaluate_model_performance(self, model: nn.Module, data_loader) -> float:
        """
        Evalúa el rendimiento de un modelo en un conjunto de datos.
        """
        if data_loader is None:
            return random.random()  # Simular rendimiento

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                total_loss += loss.item()

        return total_loss


class DynamicWeightFusionOptimizer(AdaptiveEnsembleOptimizer):
    """
    Optimizador de ensemble que fusiona dinámicamente los pesos de múltiples modelos
    basándose en su rendimiento individual y la diversidad del ensemble.
    """

    def __init__(self, fusion_method: str = "performance_weighted",
                 diversity_weight: float = 0.3, config=None):
        super().__init__(config)
        self.fusion_method = self.config.get('fusion_method', fusion_method)
        self.diversity_weight = self.config.get('diversity_weight', diversity_weight)
        logger.info(f"DynamicWeightFusionOptimizer inicializado: method={self.fusion_method}, diversity_weight={self.diversity_weight}")

    def _calculate_model_performance_weights(self, ensemble_models: List[nn.Module], data_loader) -> List[float]:
        """
        Calcula los pesos de rendimiento para cada modelo en el ensemble.
        """
        performances = []
        for model in ensemble_models:
            performance = self._evaluate_model_performance(model, data_loader)
            performances.append(performance)

        # Normalizar performances para que sumen 1
        total_performance = sum(performances)
        if total_performance > 0:
            performance_weights = [p / total_performance for p in performances]
        else:
            performance_weights = [1.0 / len(performances)] * len(performances)

        return performance_weights

    def _calculate_diversity_weights(self, ensemble_models: List[nn.Module]) -> List[float]:
        """
        Calcula los pesos de diversidad para cada modelo en el ensemble.
        """
        diversity_scores = []

        for i, model in enumerate(ensemble_models):
            diversity_score = 0.0

            # Calcular diversidad con respecto a otros modelos
            for j, other_model in enumerate(ensemble_models):
                if i != j:
                    # Calcular distancia entre pesos
                    distance = self._calculate_weight_distance(model, other_model)
                    diversity_score += distance

            diversity_scores.append(diversity_score)

        # Normalizar scores de diversidad
        total_diversity = sum(diversity_scores)
        if total_diversity > 0:
            diversity_weights = [d / total_diversity for d in diversity_scores]
        else:
            diversity_weights = [1.0 / len(ensemble_models)] * len(ensemble_models)

        return diversity_weights

    def _calculate_weight_distance(self, model1: nn.Module, model2: nn.Module) -> float:
        """
        Calcula la distancia entre los pesos de dos modelos.
        """
        total_distance = 0.0
        total_params = 0

        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 == name2:
                distance = torch.norm(param1 - param2).item()
                total_distance += distance
                total_params += 1

        return total_distance / total_params if total_params > 0 else 0.0

    def _fuse_weights(self, ensemble_models: List[nn.Module], weights: List[float]) -> nn.Module:
        """
        Fusiona los pesos de los modelos del ensemble usando los pesos especificados.
        """
        if not ensemble_models:
            raise ValueError("La lista de modelos del ensemble no puede estar vacía.")

        # Crear modelo base para la fusión
        fused_model = copy.deepcopy(ensemble_models[0])

        # Fusionar pesos
        for name, param in fused_model.named_parameters():
            fused_weight = torch.zeros_like(param)

            for i, model in enumerate(ensemble_models):
                if name in dict(model.named_parameters()):
                    model_weight = dict(model.named_parameters())[name]
                    fused_weight += weights[i] * model_weight

            param.data = fused_weight

        return fused_model

    def optimize_ensemble_weights(self, ensemble_models: List[nn.Module], data_loader=None) -> List[nn.Module]:
        logger.info("Iniciando optimización de ensemble con fusión dinámica de pesos.")

        if not ensemble_models:
            logger.warning("No hay modelos en el ensemble para optimizar.")
            return []

        # Calcular pesos de rendimiento
        performance_weights = self._calculate_model_performance_weights(ensemble_models, data_loader)

        # Calcular pesos de diversidad
        diversity_weights = self._calculate_diversity_weights(ensemble_models)

        # Combinar pesos de rendimiento y diversidad
        combined_weights = []
        for i in range(len(ensemble_models)):
            combined_weight = (1 - self.diversity_weight) * performance_weights[i] + \
                self.diversity_weight * diversity_weights[i]
            combined_weights.append(combined_weight)

        # Normalizar pesos combinados
        total_weight = sum(combined_weights)
        if total_weight > 0:
            combined_weights = [w / total_weight for w in combined_weights]

        # Fusionar pesos
        fused_model = self._fuse_weights(ensemble_models, combined_weights)

        logger.info("Optimización de ensemble con fusión dinámica completada.")
        return [fused_model]


class AdaptiveBoostingOptimizer(AdaptiveEnsembleOptimizer):
    """
    Optimizador de ensemble basado en AdaBoost que ajusta dinámicamente
    los pesos de los modelos basándose en su rendimiento en muestras difíciles.
    """

    def __init__(self, max_iterations: int = 10, learning_rate: float = 1.0, config=None):
        super().__init__(config)
        self.max_iterations = self.config.get('max_iterations', max_iterations)
        self.learning_rate = self.config.get('learning_rate', learning_rate)
        logger.info(f"AdaptiveBoostingOptimizer inicializado: max_iter={self.max_iterations}, lr={self.learning_rate}")

    def _calculate_sample_weights(self, ensemble_models: List[nn.Module], data_loader) -> List[float]:
        """
        Calcula los pesos de las muestras basándose en el rendimiento del ensemble.
        """
        if data_loader is None:
            return [1.0] * 100  # Simular pesos de muestras

        sample_weights = []

        for inputs, targets in data_loader:
            # Calcular predicción del ensemble
            ensemble_outputs = torch.zeros_like(targets)

            for model in ensemble_models:
                model.eval()
                with torch.no_grad():
                    model_outputs = model(inputs)
                    ensemble_outputs += model_outputs

            ensemble_outputs /= len(ensemble_models)

            # Calcular error para cada muestra
            errors = torch.abs(ensemble_outputs - targets)

            # Convertir errores a pesos (muestras con mayor error tienen mayor peso)
            max_error = torch.max(errors)
            if max_error > 0:
                weights = errors / max_error
            else:
                weights = torch.ones_like(errors)

            sample_weights.extend(weights.flatten().tolist())

        return sample_weights

    def _update_model_weights(self, ensemble_models: List[nn.Module],
                              sample_weights: List[float], data_loader) -> List[float]:
        """
        Actualiza los pesos de los modelos basándose en los pesos de las muestras.
        """
        model_weights = []

        for model in ensemble_models:
            # Calcular error ponderado del modelo
            weighted_error = 0.0
            total_weight = 0.0

            if data_loader is not None:
                for inputs, targets in data_loader:
                    model.eval()
                    with torch.no_grad():
                        outputs = model(inputs)
                        errors = torch.abs(outputs - targets)

                        # Aplicar pesos de muestras
                        weighted_errors = errors * torch.tensor(sample_weights[:len(errors)])
                        weighted_error += torch.sum(weighted_errors).item()
                        total_weight += torch.sum(torch.tensor(sample_weights[:len(errors)])).item()

            # Calcular peso del modelo
            if total_weight > 0:
                error_rate = weighted_error / total_weight
                if error_rate < 0.5:  # Solo considerar modelos con error < 50%
                    model_weight = self.learning_rate * np.log((1 - error_rate) / error_rate)
                else:
                    model_weight = 0.0
            else:
                model_weight = 1.0

            model_weights.append(model_weight)

        # Normalizar pesos de modelos
        total_weight = sum(model_weights)
        if total_weight > 0:
            model_weights = [w / total_weight for w in model_weights]

        return model_weights

    def optimize_ensemble_weights(self, ensemble_models: List[nn.Module], data_loader=None) -> List[nn.Module]:
        logger.info("Iniciando optimización de ensemble con AdaBoost adaptativo.")

        if not ensemble_models:
            logger.warning("No hay modelos en el ensemble para optimizar.")
            return []

        # Inicializar pesos de modelos
        model_weights = [1.0 / len(ensemble_models)] * len(ensemble_models)

        # Iteraciones de AdaBoost
        for iteration in range(self.max_iterations):
            logger.debug(f"Iteración AdaBoost {iteration + 1}/{self.max_iterations}")

            # Calcular pesos de muestras
            sample_weights = self._calculate_sample_weights(ensemble_models, data_loader)

            # Actualizar pesos de modelos
            model_weights = self._update_model_weights(ensemble_models, sample_weights, data_loader)

            # Aplicar pesos a los modelos
            for i, model in enumerate(ensemble_models):
                for param in model.parameters():
                    param.data *= model_weights[i]

            logger.debug(f"Iteración {iteration + 1}: Pesos de modelos = {model_weights}")

        logger.info("Optimización de ensemble con AdaBoost adaptativo completada.")
        return ensemble_models


class StackingEnsembleOptimizer(AdaptiveEnsembleOptimizer):
    """
    Optimizador de ensemble basado en stacking que entrena un meta-modelo
    para combinar las predicciones de los modelos base de manera óptima.
    """

    def __init__(self, meta_model_hidden_size: int = 64, meta_learning_rate: float = 0.001,
                 meta_epochs: int = 50, config=None):
        super().__init__(config)
        self.meta_model_hidden_size = self.config.get('meta_model_hidden_size', meta_model_hidden_size)
        self.meta_learning_rate = self.config.get('meta_learning_rate', meta_learning_rate)
        self.meta_epochs = self.config.get('meta_epochs', meta_epochs)
        logger.info(f"StackingEnsembleOptimizer inicializado: hidden_size={self.meta_model_hidden_size}, lr={self.meta_learning_rate}, epochs={self.meta_epochs}")

    def _create_meta_model(self, num_base_models: int) -> nn.Module:
        """
        Crea el meta-modelo para combinar las predicciones de los modelos base.
        """
        meta_model = nn.Sequential(
            nn.Linear(num_base_models, self.meta_model_hidden_size),
            nn.ReLU(),
            nn.Linear(self.meta_model_hidden_size, self.meta_model_hidden_size),
            nn.ReLU(),
            nn.Linear(self.meta_model_hidden_size, 1)
        )

        return meta_model

    def _generate_meta_features(self, ensemble_models: List[nn.Module], data_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera características meta (predicciones de modelos base) y etiquetas.
        """
        meta_features = []
        meta_labels = []

        if data_loader is not None:
            for inputs, targets in data_loader:
                # Generar predicciones de modelos base
                base_predictions = []
                for model in ensemble_models:
                    model.eval()
                    with torch.no_grad():
                        predictions = model(inputs)
                        base_predictions.append(predictions)

                # Combinar predicciones
                combined_predictions = torch.cat(base_predictions, dim=1)
                meta_features.append(combined_predictions)
                meta_labels.append(targets)

        if meta_features:
            meta_features = torch.cat(meta_features, dim=0)
            meta_labels = torch.cat(meta_labels, dim=0)
        else:
            # Crear datos simulados
            num_samples = 100
            meta_features = torch.randn(num_samples, len(ensemble_models))
            meta_labels = torch.randn(num_samples, 1)

        return meta_features, meta_labels

    def _train_meta_model(self, meta_model: nn.Module, meta_features: torch.Tensor,
                          meta_labels: torch.Tensor) -> nn.Module:
        """
        Entrena el meta-modelo para combinar las predicciones de los modelos base.
        """
        optimizer = optim.Adam(meta_model.parameters(), lr=self.meta_learning_rate)
        criterion = nn.MSELoss()

        meta_model.train()

        for epoch in range(self.meta_epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = meta_model(meta_features)
            loss = criterion(outputs, meta_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Meta-modelo epoch {epoch}: Loss = {loss.item():.6f}")

        return meta_model

    def optimize_ensemble_weights(self, ensemble_models: List[nn.Module], data_loader=None) -> List[nn.Module]:
        logger.info("Iniciando optimización de ensemble con stacking adaptativo.")

        if not ensemble_models:
            logger.warning("No hay modelos en el ensemble para optimizar.")
            return []

        # Crear meta-modelo
        meta_model = self._create_meta_model(len(ensemble_models))

        # Generar características meta
        meta_features, meta_labels = self._generate_meta_features(ensemble_models, data_loader)

        # Entrenar meta-modelo
        trained_meta_model = self._train_meta_model(meta_model, meta_features, meta_labels)

        # Aplicar pesos del meta-modelo a los modelos base
        with torch.no_grad():
            # Obtener pesos de la primera capa del meta-modelo
            meta_weights = trained_meta_model[0].weight.data[0]  # Primer neurona de salida

            # Normalizar pesos
            meta_weights = torch.softmax(meta_weights, dim=0)

            # Aplicar pesos a los modelos base
            for i, model in enumerate(ensemble_models):
                weight_factor = meta_weights[i].item()
                for param in model.parameters():
                    param.data *= weight_factor

        logger.info("Optimización de ensemble con stacking adaptativo completada.")
        return ensemble_models


class EnsemblePerformanceAnalyzer:
    """
    Analizador para evaluar el rendimiento de los ensembles adaptativos.
    """

    def __init__(self):
        logger.info("EnsemblePerformanceAnalyzer inicializado.")

    def analyze_ensemble_performance(self, ensemble_models: List[nn.Module],
                                     test_data_loader) -> Dict[str, float]:
        """
        Analiza el rendimiento del ensemble.
        """
        analysis_results = {}

        if not ensemble_models:
            logger.warning("No hay modelos en el ensemble para analizar.")
            return analysis_results

        # Evaluar rendimiento individual de cada modelo
        individual_performances = []
        for i, model in enumerate(ensemble_models):
            performance = self._evaluate_model_performance(model, test_data_loader)
            individual_performances.append(performance)
            analysis_results[f'model_{i}_performance'] = performance

        # Evaluar rendimiento del ensemble
        ensemble_performance = self._evaluate_ensemble_performance(ensemble_models, test_data_loader)
        analysis_results['ensemble_performance'] = ensemble_performance

        # Calcular diversidad del ensemble
        diversity = self._calculate_ensemble_diversity(ensemble_models)
        analysis_results['ensemble_diversity'] = diversity

        # Calcular mejora del ensemble sobre el promedio individual
        average_individual_performance = np.mean(individual_performances)
        ensemble_improvement = average_individual_performance - ensemble_performance
        ensemble_improvement_percentage = (ensemble_improvement / average_individual_performance) * 100

        analysis_results['average_individual_performance'] = average_individual_performance
        analysis_results['ensemble_improvement'] = ensemble_improvement
        analysis_results['ensemble_improvement_percentage'] = ensemble_improvement_percentage

        logger.info(f"Análisis de ensemble: Mejora = {ensemble_improvement_percentage:.2f}%, Diversidad = {diversity:.4f}")
        return analysis_results

    def _evaluate_model_performance(self, model: nn.Module, data_loader) -> float:
        """
        Evalúa el rendimiento de un modelo individual.
        """
        if data_loader is None:
            return random.random()

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
                total_loss += loss.item()

        return total_loss

    def _evaluate_ensemble_performance(self, ensemble_models: List[nn.Module], data_loader) -> float:
        """
        Evalúa el rendimiento del ensemble combinando las predicciones.
        """
        if data_loader is None:
            return random.random()

        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                # Combinar predicciones del ensemble
                ensemble_outputs = torch.zeros_like(targets)

                for model in ensemble_models:
                    model.eval()
                    model_outputs = model(inputs)
                    ensemble_outputs += model_outputs

                ensemble_outputs /= len(ensemble_models)

                # Calcular pérdida
                loss = nn.functional.mse_loss(ensemble_outputs, targets)
                total_loss += loss.item()

        return total_loss

    def _calculate_ensemble_diversity(self, ensemble_models: List[nn.Module]) -> float:
        """
        Calcula la diversidad del ensemble basándose en la distancia entre modelos.
        """
        if len(ensemble_models) < 2:
            return 0.0

        total_distance = 0.0
        num_pairs = 0

        for i in range(len(ensemble_models)):
            for j in range(i + 1, len(ensemble_models)):
                distance = self._calculate_weight_distance(ensemble_models[i], ensemble_models[j])
                total_distance += distance
                num_pairs += 1

        return total_distance / num_pairs if num_pairs > 0 else 0.0

    def _calculate_weight_distance(self, model1: nn.Module, model2: nn.Module) -> float:
        """
        Calcula la distancia entre los pesos de dos modelos.
        """
        total_distance = 0.0
        total_params = 0

        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 == name2:
                distance = torch.norm(param1 - param2).item()
                total_distance += distance
                total_params += 1

        return total_distance / total_params if total_params > 0 else 0.0


def create_adaptive_ensemble_optimizer(optimizer_type: str, **kwargs) -> AdaptiveEnsembleOptimizer:
    """
    Factoría para crear diferentes tipos de optimizadores de ensemble adaptativos.
    """
    if optimizer_type == "dynamic_fusion":
        return DynamicWeightFusionOptimizer(**kwargs)
    elif optimizer_type == "adaptive_boosting":
        return AdaptiveBoostingOptimizer(**kwargs)
    elif optimizer_type == "stacking":
        return StackingEnsembleOptimizer(**kwargs)
    else:
        raise ValueError(f"Tipo de optimizador de ensemble adaptativo no soportado: {optimizer_type}")


def optimize_ensemble_weights(ensemble_models: List[nn.Module], optimizer_type: str,
                              data_loader=None, **kwargs) -> Tuple[List[nn.Module], Dict[str, float]]:
    """
    Función de conveniencia para optimizar los pesos de un ensemble de modelos.
    """
    optimizer = create_adaptive_ensemble_optimizer(optimizer_type, **kwargs)
    original_ensemble = [copy.deepcopy(model) for model in ensemble_models]

    # Optimizar ensemble
    optimized_ensemble = optimizer.optimize_ensemble_weights(ensemble_models, data_loader)

    # Analizar rendimiento
    analyzer = EnsemblePerformanceAnalyzer()
    analysis = analyzer.analyze_ensemble_performance(optimized_ensemble, data_loader)

    return optimized_ensemble, analysis


# Exportar clases y funciones principales
__all__ = [
    'AdaptiveEnsembleOptimizer',
    'DynamicWeightFusionOptimizer',
    'AdaptiveBoostingOptimizer',
    'StackingEnsembleOptimizer',
    'EnsemblePerformanceAnalyzer',
    'create_adaptive_ensemble_optimizer',
    'optimize_ensemble_weights'
]

logger.info("RFEN5_RN_8 - Ensembles Adaptativos para Optimización de Pesos cargado correctamente")
