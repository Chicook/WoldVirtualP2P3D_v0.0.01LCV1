try:
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as prune
except ImportError:
    pass  # dependencia pesada opcional
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import random

# Configuración del logger
logger = logging.getLogger(__name__)


class IntelligentWeightPruner(ABC):
    """
    Clase base abstracta para sistemas de poda inteligente avanzada de pesos.
    Define la interfaz común para todas las estrategias de poda inteligente.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else {}
        logger.info("IntelligentWeightPruner base inicializado.")

    @abstractmethod
    def intelligent_prune(self, model: nn.Module, data_loader=None) -> nn.Module:
        """
        Método abstracto para aplicar poda inteligente a los pesos del modelo.
        Debe ser implementado por las subclases.
        """
        pass

    def _calculate_weight_importance(self, model: nn.Module, data_loader=None) -> Dict[str, torch.Tensor]:
        """
        Calcula la importancia de cada peso en el modelo.
        """
        importance_scores = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Calcular importancia basada en múltiples criterios
                weights = module.weight.data

                # Criterio 1: Magnitud absoluta
                magnitude_importance = torch.abs(weights)

                # Criterio 2: Gradiente (si está disponible)
                gradient_importance = torch.zeros_like(weights)
                if weights.grad is not None:
                    gradient_importance = torch.abs(weights.grad)

                # Criterio 3: Sensibilidad (simulada)
                sensitivity_importance = torch.ones_like(weights) * 0.5

                # Combinar criterios con pesos
                combined_importance = (
                    0.4 * magnitude_importance +
                    0.4 * gradient_importance +
                    0.2 * sensitivity_importance
                )

                importance_scores[name] = combined_importance
                logger.debug(f"Importancia calculada para la capa {name}.")

        return importance_scores


class AdaptiveMagnitudePruner(IntelligentWeightPruner):
    """
    Sistema de poda por magnitud adaptativa que ajusta dinámicamente
    el umbral de poda basado en la distribución de pesos de cada capa.
    """

    def __init__(self, base_sparsity: float = 0.3, adaptation_factor: float = 0.1, config=None):
        super().__init__(config)
        self.base_sparsity = self.config.get('base_sparsity', base_sparsity)
        self.adaptation_factor = self.config.get('adaptation_factor', adaptation_factor)
        logger.info(f"AdaptiveMagnitudePruner inicializado con sparsity={self.base_sparsity}, adaptation={self.adaptation_factor}")

    def _calculate_adaptive_threshold(self, weights: torch.Tensor) -> float:
        """
        Calcula un umbral adaptativo basado en la distribución de pesos.
        """
        # Calcular estadísticas de la distribución
        mean_weight = torch.mean(torch.abs(weights))
        std_weight = torch.std(torch.abs(weights))

        # Umbral adaptativo basado en percentiles
        threshold = torch.quantile(torch.abs(weights), self.base_sparsity)

        # Ajustar basado en la desviación estándar
        adaptive_threshold = threshold + (std_weight * self.adaptation_factor)

        return adaptive_threshold.item()

    def intelligent_prune(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando poda inteligente adaptativa por magnitud.")

        importance_scores = self._calculate_weight_importance(model, data_loader)

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weights = module.weight.data

                # Calcular umbral adaptativo para esta capa
                adaptive_threshold = self._calculate_adaptive_threshold(weights)

                # Crear máscara de poda basada en el umbral adaptativo
                mask = torch.abs(weights) > adaptive_threshold

                # Aplicar poda
                module.weight.data = weights * mask.float()

                # Calcular sparsity real
                actual_sparsity = 1.0 - (mask.sum().item() / mask.numel())
                logger.debug(f"Capa {name}: Sparsity={actual_sparsity:.3f}, Umbral={adaptive_threshold:.6f}")

        logger.info("Poda inteligente adaptativa por magnitud completada.")
        return model


class GradientBasedPruner(IntelligentWeightPruner):
    """
    Sistema de poda basado en gradientes que considera tanto la magnitud
    de los pesos como la sensibilidad de los gradientes.
    """

    def __init__(self, gradient_threshold: float = 0.01, magnitude_weight: float = 0.6, config=None):
        super().__init__(config)
        self.gradient_threshold = self.config.get('gradient_threshold', gradient_threshold)
        self.magnitude_weight = self.config.get('magnitude_weight', magnitude_weight)
        self.gradient_weight = 1.0 - self.magnitude_weight
        logger.info(f"GradientBasedPruner inicializado con grad_threshold={self.gradient_threshold}, mag_weight={self.magnitude_weight}")

    def _calculate_gradient_importance(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Calcula la importancia basada en gradientes.
        """
        gradient_importance = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Calcular importancia del gradiente
                grad_magnitude = torch.abs(param.grad)
                gradient_importance[name] = grad_magnitude
                logger.debug(f"Importancia de gradiente calculada para {name}.")

        return gradient_importance

    def intelligent_prune(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando poda inteligente basada en gradientes.")

        # Calcular importancia de gradientes
        gradient_importance = self._calculate_gradient_importance(model)

        # Calcular importancia de magnitud
        magnitude_importance = self._calculate_weight_importance(model, data_loader)

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weights = module.weight.data

                # Obtener importancia combinada
                mag_imp = magnitude_importance.get(name, torch.ones_like(weights))
                grad_imp = gradient_importance.get(name, torch.zeros_like(weights))

                # Combinar importancia de magnitud y gradiente
                combined_importance = (
                    self.magnitude_weight * mag_imp +
                    self.gradient_weight * grad_imp
                )

                # Crear máscara basada en la importancia combinada
                threshold = torch.quantile(combined_importance, self.gradient_threshold)
                mask = combined_importance > threshold

                # Aplicar poda
                module.weight.data = weights * mask.float()

                logger.debug(f"Capa {name}: Poda basada en gradiente aplicada.")

        logger.info("Poda inteligente basada en gradientes completada.")
        return model


class StructuredIntelligentPruner(IntelligentWeightPruner):
    """
    Sistema de poda estructurada inteligente que elimina canales enteros
    o neuronas basándose en su contribución al rendimiento del modelo.
    """

    def __init__(self, channel_sparsity: float = 0.2, neuron_sparsity: float = 0.3, config=None):
        super().__init__(config)
        self.channel_sparsity = self.config.get('channel_sparsity', channel_sparsity)
        self.neuron_sparsity = self.config.get('neuron_sparsity', neuron_sparsity)
        logger.info(f"StructuredIntelligentPruner inicializado con channel_sparsity={self.channel_sparsity}, neuron_sparsity={self.neuron_sparsity}")

    def _calculate_channel_importance(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Calcula la importancia de cada canal (filtro) en una capa convolucional.
        """
        if len(weights.shape) == 4:  # Conv2d: [out_channels, in_channels, height, width]
            # Calcular la norma L2 de cada canal de salida
            channel_importance = torch.norm(weights.view(weights.shape[0], -1), dim=1)
        elif len(weights.shape) == 2:  # Linear: [out_features, in_features]
            # Para capas lineales, calcular importancia por neurona de salida
            channel_importance = torch.norm(weights, dim=1)
        else:
            channel_importance = torch.ones(weights.shape[0])

        return channel_importance

    def intelligent_prune(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando poda estructurada inteligente.")

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Poda de canales para capas convolucionales
                weights = module.weight.data
                channel_importance = self._calculate_channel_importance(weights)

                # Determinar canales a podar
                threshold = torch.quantile(channel_importance, self.channel_sparsity)
                channels_to_keep = channel_importance > threshold

                # Aplicar poda estructurada
                new_weights = weights[channels_to_keep]
                module.weight.data = new_weights

                logger.debug(f"Capa convolucional {name}: {channels_to_keep.sum().item()}/{len(channels_to_keep)} canales mantenidos.")

            elif isinstance(module, nn.Linear):
                # Poda de neuronas para capas lineales
                weights = module.weight.data
                neuron_importance = self._calculate_channel_importance(weights)

                # Determinar neuronas a podar
                threshold = torch.quantile(neuron_importance, self.neuron_sparsity)
                neurons_to_keep = neuron_importance > threshold

                # Aplicar poda estructurada
                new_weights = weights[neurons_to_keep]
                module.weight.data = new_weights

                logger.debug(f"Capa lineal {name}: {neurons_to_keep.sum().item()}/{len(neurons_to_keep)} neuronas mantenidas.")

        logger.info("Poda estructurada inteligente completada.")
        return model


class DynamicPruningScheduler(IntelligentWeightPruner):
    """
    Sistema de programación dinámica de poda que ajusta la intensidad
    de poda a lo largo del entrenamiento basándose en el rendimiento.
    """

    def __init__(self, initial_sparsity: float = 0.1, final_sparsity: float = 0.8,
                 schedule_type: str = "polynomial", config=None):
        super().__init__(config)
        self.initial_sparsity = self.config.get('initial_sparsity', initial_sparsity)
        self.final_sparsity = self.config.get('final_sparsity', final_sparsity)
        self.schedule_type = self.config.get('schedule_type', schedule_type)
        self.current_step = 0
        self.total_steps = self.config.get('total_steps', 1000)
        logger.info(f"DynamicPruningScheduler inicializado: {initial_sparsity} -> {final_sparsity} ({schedule_type})")

    def _calculate_current_sparsity(self) -> float:
        """
        Calcula la sparsity actual basándose en el paso de entrenamiento.
        """
        progress = min(self.current_step / self.total_steps, 1.0)

        if self.schedule_type == "linear":
            current_sparsity = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress
        elif self.schedule_type == "polynomial":
            current_sparsity = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * (progress ** 2)
        elif self.schedule_type == "exponential":
            current_sparsity = self.initial_sparsity * ((self.final_sparsity / self.initial_sparsity) ** progress)
        else:
            current_sparsity = self.initial_sparsity

        return current_sparsity

    def intelligent_prune(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando poda dinámica programada.")

        # Calcular sparsity actual
        current_sparsity = self._calculate_current_sparsity()
        logger.debug(f"Paso {self.current_step}: Sparsity actual = {current_sparsity:.3f}")

        # Aplicar poda con la sparsity actual
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weights = module.weight.data

                # Crear máscara basada en la sparsity actual
                threshold = torch.quantile(torch.abs(weights), current_sparsity)
                mask = torch.abs(weights) > threshold

                # Aplicar poda
                module.weight.data = weights * mask.float()

                logger.debug(f"Capa {name}: Poda aplicada con sparsity {current_sparsity:.3f}")

        # Incrementar paso
        self.current_step += 1

        logger.info("Poda dinámica programada completada.")
        return model

    def update_schedule(self, current_step: int, total_steps: int):
        """
        Actualiza los parámetros del programador de poda.
        """
        self.current_step = current_step
        self.total_steps = total_steps
        logger.debug(f"Programador de poda actualizado: paso {current_step}/{total_steps}")


class PruningAnalyzer:
    """
    Analizador para evaluar el impacto de la poda en el modelo.
    """

    def __init__(self):
        logger.info("PruningAnalyzer inicializado.")

    def analyze_pruning_impact(self, original_model: nn.Module, pruned_model: nn.Module) -> Dict[str, float]:
        """
        Analiza el impacto de la poda comparando el modelo original con el podado.
        """
        analysis_results = {}

        # Contar parámetros
        original_params = sum(p.numel() for p in original_model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())

        analysis_results['parameter_reduction'] = 1.0 - (pruned_params / original_params)
        analysis_results['original_parameters'] = original_params
        analysis_results['pruned_parameters'] = pruned_params

        # Calcular sparsity por capa
        layer_sparsity = {}
        for name, module in original_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                original_weights = module.weight.data
                pruned_weights = pruned_model.state_dict().get(f"{name}.weight", original_weights)

                # Calcular sparsity
                zero_count = (pruned_weights == 0).sum().item()
                total_count = pruned_weights.numel()
                sparsity = zero_count / total_count

                layer_sparsity[name] = sparsity

        analysis_results['layer_sparsity'] = layer_sparsity
        analysis_results['average_sparsity'] = np.mean(list(layer_sparsity.values()))

        logger.info(f"Análisis de poda completado: {analysis_results['parameter_reduction']*100:.2f}% reducción de parámetros")
        return analysis_results


def create_intelligent_pruner(pruner_type: str, **kwargs) -> IntelligentWeightPruner:
    """
    Factoría para crear diferentes tipos de podadores inteligentes.
    """
    if pruner_type == "adaptive_magnitude":
        return AdaptiveMagnitudePruner(**kwargs)
    elif pruner_type == "gradient_based":
        return GradientBasedPruner(**kwargs)
    elif pruner_type == "structured":
        return StructuredIntelligentPruner(**kwargs)
    elif pruner_type == "dynamic_schedule":
        return DynamicPruningScheduler(**kwargs)
    else:
        raise ValueError(f"Tipo de podador inteligente no soportado: {pruner_type}")


def intelligent_prune_model(model: nn.Module, pruner_type: str, data_loader=None, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Función de conveniencia para aplicar poda inteligente a un modelo.
    """
    pruner = create_intelligent_pruner(pruner_type, **kwargs)
    original_model = model

    # Aplicar poda
    pruned_model = pruner.intelligent_prune(model, data_loader)

    # Analizar impacto
    analyzer = PruningAnalyzer()
    analysis = analyzer.analyze_pruning_impact(original_model, pruned_model)

    return pruned_model, analysis


# Exportar clases y funciones principales
__all__ = [
    'IntelligentWeightPruner',
    'AdaptiveMagnitudePruner',
    'GradientBasedPruner',
    'StructuredIntelligentPruner',
    'DynamicPruningScheduler',
    'PruningAnalyzer',
    'create_intelligent_pruner',
    'intelligent_prune_model'
]

logger.info("RFEN5_RN_5 - Poda Inteligente Avanzada de Pesos cargada correctamente")
