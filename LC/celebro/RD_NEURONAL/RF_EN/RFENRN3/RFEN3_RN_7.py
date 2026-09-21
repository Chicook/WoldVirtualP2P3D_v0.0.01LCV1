"""
RFEN3_RN_7 - Sistema de Aprendizaje Transferido Avanzado
Implementación de técnicas modernas de transfer learning y fine-tuning
Incluye: Feature extraction, domain adaptation, multi-task learning, y knowledge distillation
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy
import pickle
import os

logger = logging.getLogger(__name__)


@dataclass
class TransferLearningConfig:
    """Configuración para aprendizaje transferido"""
    source_model_path: str = ""
    target_task: str = ""
    transfer_strategy: str = "feature_extraction"  # feature_extraction, fine_tuning, multi_task
    freeze_layers: int = -1  # -1 significa no congelar, 0 significa congelar todo excepto la última capa
    learning_rate_multiplier: float = 0.1  # Multiplicador para capas no congeladas
    domain_adaptation: bool = False
    knowledge_distillation: bool = False
    multi_task_weighting: str = "equal"  # equal, uncertainty, gradient_norm
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7


class BaseTransferLearner(ABC):
    """Clase base abstracta para métodos de transfer learning"""

    def __init__(self, config: TransferLearningConfig):
        self.config = config
        self.source_model = None
        self.target_model = None
        self.transfer_stats = {}

    @abstractmethod
    def transfer_knowledge(self, source_model: nn.Module, target_model: nn.Module) -> nn.Module:
        """Transfiere conocimiento del modelo fuente al modelo objetivo"""
        pass

    @abstractmethod
    def adapt_to_target_task(self, model: nn.Module, target_data: torch.Tensor) -> nn.Module:
        """Adapta el modelo a la tarea objetivo"""
        pass


class FeatureExtractor(BaseTransferLearner):
    """
    Extractores de características para transfer learning
    """

    def __init__(self, config: TransferLearningConfig):
        super().__init__(config)
        self.feature_maps = {}
        self.feature_importance = {}

    def transfer_knowledge(self, source_model: nn.Module, target_model: nn.Module) -> nn.Module:
        """Transfiere características del modelo fuente"""

        logger.info("Iniciando transferencia de características")

        # Copiar pesos de capas compatibles
        transferred_layers = 0

        for (src_name, src_param), (tgt_name, tgt_param) in zip(
            source_model.named_parameters(), target_model.named_parameters()
        ):
            if src_param.shape == tgt_param.shape:
                tgt_param.data.copy_(src_param.data)
                transferred_layers += 1

                # Registrar importancia de la característica
                self.feature_importance[tgt_name] = torch.norm(src_param.data).item()

        # Congelar capas según configuración
        self._freeze_layers(target_model)

        self.transfer_stats['transferred_layers'] = transferred_layers
        logger.info(f"Transferidas {transferred_layers} capas")

        return target_model

    def adapt_to_target_task(self, model: nn.Module, target_data: torch.Tensor) -> nn.Module:
        """Adapta las características extraídas a la tarea objetivo"""

        # Analizar características extraídas
        with torch.no_grad():
            features = self._extract_features(model, target_data)
            self.feature_maps['target_features'] = features

        # Calcular adaptabilidad de características
        adaptability_scores = self._calculate_feature_adaptability(features)

        # Ajustar pesos de características menos adaptables
        self._adjust_feature_weights(model, adaptability_scores)

        return model

    def _freeze_layers(self, model: nn.Module) -> None:
        """Congela capas según la configuración"""

        if self.config.freeze_layers == -1:
            return  # No congelar ninguna capa

        layers = list(model.named_parameters())

        for i, (name, param) in enumerate(layers):
            if i < len(layers) - self.config.freeze_layers:
                param.requires_grad = False
                logger.debug(f"Capa congelada: {name}")

    def _extract_features(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Extrae características del modelo"""

        features = []

        def hook_fn(module, input, output):
            features.append(output.detach())

        # Registrar hooks en capas intermedias
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = model(data)

        # Remover hooks
        for hook in hooks:
            hook.remove()

        return torch.cat([f.flatten(1) for f in features], dim=1)

    def _calculate_feature_adaptability(self, features: torch.Tensor) -> torch.Tensor:
        """Calcula la adaptabilidad de las características"""

        # Calcular varianza de características
        feature_variance = torch.var(features, dim=0)

        # Calcular correlación entre características
        feature_corr = torch.corrcoef(features.T)

        # Adaptabilidad = alta varianza + baja correlación
        adaptability = feature_variance / (1 + torch.mean(torch.abs(feature_corr), dim=1))

        return adaptability

    def _adjust_feature_weights(self, model: nn.Module, adaptability_scores: torch.Tensor) -> None:
        """Ajusta pesos de características basado en adaptabilidad"""

        score_idx = 0

        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                # Aplicar multiplicador basado en adaptabilidad
                if score_idx < len(adaptability_scores):
                    multiplier = adaptability_scores[score_idx].item()
                    param.data *= (1 + multiplier * 0.1)  # Ajuste suave
                    score_idx += 1


class DomainAdapter(BaseTransferLearner):
    """
    Adaptador de dominio para transfer learning entre dominios diferentes
    """

    def __init__(self, config: TransferLearningConfig):
        super().__init__(config)
        self.domain_classifier = None
        self.gradient_reversal_layer = None
        self.domain_loss_weight = 0.1

    def transfer_knowledge(self, source_model: nn.Module, target_model: nn.Module) -> nn.Module:
        """Transfiere conocimiento con adaptación de dominio"""

        logger.info("Iniciando transferencia con adaptación de dominio")

        # Copiar arquitectura base
        target_model.load_state_dict(source_model.state_dict(), strict=False)

        # Crear clasificador de dominio
        self._create_domain_classifier(target_model)

        # Crear capa de reversión de gradientes
        self.gradient_reversal_layer = GradientReversalLayer(self.domain_loss_weight)

        return target_model

    def adapt_to_target_task(self, model: nn.Module, target_data: torch.Tensor) -> nn.Module:
        """Adapta el modelo al dominio objetivo"""

        # Crear datos sintéticos del dominio fuente si es necesario
        source_data = self._generate_source_domain_data(target_data)

        # Entrenar adaptador de dominio
        self._train_domain_adapter(model, source_data, target_data)

        return model

    def _create_domain_classifier(self, model: nn.Module) -> None:
        """Crea clasificador de dominio"""

        # Obtener tamaño de características
        feature_size = self._get_feature_size(model)

        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 2 dominios: fuente y objetivo
        )

    def _get_feature_size(self, model: nn.Module) -> int:
        """Obtiene el tamaño de las características del modelo"""

        # Encontrar la última capa antes de la clasificación
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                return module.in_features

        return 512  # Default

    def _generate_source_domain_data(self, target_data: torch.Tensor) -> torch.Tensor:
        """Genera datos sintéticos del dominio fuente"""

        # Usar estadísticas del dominio objetivo para generar datos fuente
        mean = torch.mean(target_data, dim=0)
        std = torch.std(target_data, dim=0)

        # Generar datos con distribución similar pero con ruido
        source_data = torch.normal(mean, std, size=target_data.shape)

        return source_data

    def _train_domain_adapter(self, model: nn.Module,
                              source_data: torch.Tensor,
                              target_data: torch.Tensor) -> None:
        """Entrena el adaptador de dominio"""

        optimizer = optim.Adam(list(model.parameters()) + list(self.domain_classifier.parameters()))

        for epoch in range(10):  # Pocas épocas para adaptación rápida

            # Crear etiquetas de dominio
            source_labels = torch.zeros(source_data.size(0), dtype=torch.long)
            target_labels = torch.ones(target_data.size(0), dtype=torch.long)

            # Combinar datos
            domain_data = torch.cat([source_data, target_data], dim=0)
            domain_labels = torch.cat([source_labels, target_labels], dim=0)

            # Forward pass
            features = self._extract_features(model, domain_data)
            domain_pred = self.domain_classifier(features)

            # Calcular pérdida de dominio
            domain_loss = F.cross_entropy(domain_pred, domain_labels)

            # Backward pass con reversión de gradientes
            optimizer.zero_grad()
            domain_loss.backward()
            optimizer.step()


class GradientReversalLayer(nn.Module):
    """
    Capa de reversión de gradientes para adversarial domain adaptation
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        return -self.alpha * grad_output


class KnowledgeDistiller(BaseTransferLearner):
    """
    Distilador de conocimiento para transfer learning
    """

    def __init__(self, config: TransferLearningConfig):
        super().__init__(config)
        self.temperature = config.distillation_temperature
        self.alpha = config.distillation_alpha

    def transfer_knowledge(self, source_model: nn.Module, target_model: nn.Module) -> nn.Module:
        """Transfiere conocimiento usando distillation"""

        logger.info("Iniciando transferencia de conocimiento por distillation")

        # El modelo objetivo ya está inicializado
        return target_model

    def adapt_to_target_task(self, model: nn.Module, target_data: torch.Tensor) -> nn.Module:
        """Adapta usando distillation de conocimiento"""

        # Crear datos de entrenamiento
        train_data = target_data[:len(target_data)//2]

        # Obtener predicciones del modelo fuente (teacher)
        with torch.no_grad():
            teacher_logits = self.source_model(train_data)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        # Entrenar modelo objetivo (student) con distillation
        self._train_with_distillation(model, train_data, teacher_probs)

        return model

    def _train_with_distillation(self, student_model: nn.Module,
                                 data: torch.Tensor,
                                 teacher_probs: torch.Tensor) -> None:
        """Entrena el modelo estudiante con distillation"""

        optimizer = optim.Adam(student_model.parameters(), lr=0.001)

        for epoch in range(20):
            optimizer.zero_grad()

            # Forward pass del estudiante
            student_logits = student_model(data)
            student_probs = F.log_softmax(student_logits / self.temperature, dim=1)

            # Calcular pérdida de distillation
            distillation_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')

            # Calcular pérdida estándar (si hay etiquetas)
            standard_loss = F.cross_entropy(student_logits, torch.argmax(teacher_probs, dim=1))

            # Pérdida combinada
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * standard_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()


class MultiTaskLearner(BaseTransferLearner):
    """
    Aprendizaje multi-tarea para transfer learning
    """

    def __init__(self, config: TransferLearningConfig):
        super().__init__(config)
        self.task_weights = {}
        self.task_losses = {}
        self.uncertainty_weights = {}

    def transfer_knowledge(self, source_model: nn.Module, target_model: nn.Module) -> nn.Module:
        """Transfiere conocimiento para aprendizaje multi-tarea"""

        logger.info("Iniciando transferencia para aprendizaje multi-tarea")

        # Crear cabezas de tarea
        self._create_task_heads(target_model)

        # Copiar backbone del modelo fuente
        self._copy_backbone(source_model, target_model)

        return target_model

    def adapt_to_target_task(self, model: nn.Module, target_data: torch.Tensor) -> nn.Module:
        """Adapta usando aprendizaje multi-tarea"""

        # Entrenar con múltiples tareas
        self._train_multi_task(model, target_data)

        return model

    def _create_task_heads(self, model: nn.Module) -> None:
        """Crea cabezas de tarea específicas"""

        # Obtener tamaño de características
        feature_size = self._get_feature_size(model)

        # Crear cabezas para diferentes tareas
        self.task_heads = nn.ModuleDict({
            'classification': nn.Linear(feature_size, 10),
            'regression': nn.Linear(feature_size, 1),
            'embedding': nn.Linear(feature_size, 128)
        })

    def _copy_backbone(self, source_model: nn.Module, target_model: nn.Module) -> None:
        """Copia el backbone del modelo fuente"""

        # Copiar todas las capas excepto la última
        source_state = source_model.state_dict()
        target_state = target_model.state_dict()

        for name, param in source_state.items():
            if name in target_state and param.shape == target_state[name].shape:
                target_state[name] = param

        target_model.load_state_dict(target_state)

    def _train_multi_task(self, model: nn.Module, data: torch.Tensor) -> None:
        """Entrena el modelo con múltiples tareas"""

        optimizer = optim.Adam(list(model.parameters()) + list(self.task_heads.parameters()))

        for epoch in range(30):
            optimizer.zero_grad()

            # Extraer características
            features = self._extract_features(model, data)

            # Calcular pérdidas para cada tarea
            task_losses = {}

            # Tarea de clasificación (simulada)
            if 'classification' in self.task_heads:
                cls_logits = self.task_heads['classification'](features)
                cls_targets = torch.randint(0, 10, (features.size(0),))
                task_losses['classification'] = F.cross_entropy(cls_logits, cls_targets)

            # Tarea de regresión (simulada)
            if 'regression' in self.task_heads:
                reg_output = self.task_heads['regression'](features)
                reg_targets = torch.randn(features.size(0), 1)
                task_losses['regression'] = F.mse_loss(reg_output, reg_targets)

            # Calcular pérdida total con pesos
            total_loss = self._calculate_weighted_loss(task_losses)

            # Backward pass
            total_loss.backward()
            optimizer.step()

    def _calculate_weighted_loss(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calcula la pérdida total ponderada"""

        if self.config.multi_task_weighting == "equal":
            return sum(task_losses.values()) / len(task_losses)

        elif self.config.multi_task_weighting == "uncertainty":
            # Ponderación basada en incertidumbre
            weights = []
            for task, loss in task_losses.items():
                uncertainty = torch.exp(-torch.log(loss + 1e-8))
                weights.append(uncertainty)

            total_weight = sum(weights)
            weighted_loss = sum(w * loss for w, loss in zip(weights, task_losses.values()))
            return weighted_loss / total_weight

        elif self.config.multi_task_weighting == "gradient_norm":
            # Ponderación basada en norma de gradientes
            weights = []
            for task, loss in task_losses.items():
                loss.backward(retain_graph=True)
                grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in self.task_heads.parameters()]))
                weights.append(1.0 / (grad_norm + 1e-8))

            total_weight = sum(weights)
            weighted_loss = sum(w * loss for w, loss in zip(weights, task_losses.values()))
            return weighted_loss / total_weight

        else:
            return sum(task_losses.values())


class TransferLearningManager:
    """
    Gestor principal de transfer learning
    Coordina todas las técnicas de transfer learning
    """

    def __init__(self, config: TransferLearningConfig):
        self.config = config
        self.transfer_methods = {
            'feature_extraction': FeatureExtractor(config),
            'domain_adaptation': DomainAdapter(config),
            'knowledge_distillation': KnowledgeDistiller(config),
            'multi_task': MultiTaskLearner(config)
        }

        self.transfer_history = []
        self.performance_metrics = {}

    def load_source_model(self, model_path: str) -> nn.Module:
        """Carga el modelo fuente"""

        if os.path.exists(model_path):
            model = torch.load(model_path, map_location='cpu')
            logger.info(f"Modelo fuente cargado desde {model_path}")
            return model
        else:
            logger.warning(f"Archivo de modelo no encontrado: {model_path}")
            return None

    def transfer_to_target_task(self, source_model: nn.Module,
                                target_model: nn.Module,
                                target_data: torch.Tensor) -> nn.Module:
        """Transfiere conocimiento a la tarea objetivo"""

        strategy = self.config.transfer_strategy

        if strategy not in self.transfer_methods:
            raise ValueError(f"Estrategia de transferencia no soportada: {strategy}")

        transfer_method = self.transfer_methods[strategy]

        # Transferir conocimiento
        transferred_model = transfer_method.transfer_knowledge(source_model, target_model)

        # Adaptar a la tarea objetivo
        adapted_model = transfer_method.adapt_to_target_task(transferred_model, target_data)

        # Registrar transferencia
        self.transfer_history.append({
            'strategy': strategy,
            'timestamp': time.time(),
            'source_model': source_model.__class__.__name__,
            'target_model': target_model.__class__.__name__
        })

        logger.info(f"Transferencia completada usando estrategia: {strategy}")

        return adapted_model

    def evaluate_transfer_performance(self, model: nn.Module,
                                      test_data: torch.Tensor,
                                      test_labels: torch.Tensor) -> Dict[str, float]:
        """Evalúa el rendimiento del modelo transferido"""

        model.eval()

        with torch.no_grad():
            predictions = model(test_data)

            if predictions.dim() > 1 and predictions.size(1) > 1:
                # Clasificación
                predicted_labels = torch.argmax(predictions, dim=1)
                accuracy = (predicted_labels == test_labels).float().mean().item()

                metrics = {
                    'accuracy': accuracy,
                    'loss': F.cross_entropy(predictions, test_labels).item()
                }
            else:
                # Regresión
                mse = F.mse_loss(predictions, test_labels.unsqueeze(1)).item()
                mae = F.l1_loss(predictions, test_labels.unsqueeze(1)).item()

                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': math.sqrt(mse)
                }

        self.performance_metrics = metrics
        return metrics

    def get_transfer_summary(self) -> Dict:
        """Obtiene un resumen del estado de transferencia"""
        return {
            'config': {
                'transfer_strategy': self.config.transfer_strategy,
                'freeze_layers': self.config.freeze_layers,
                'learning_rate_multiplier': self.config.learning_rate_multiplier,
                'domain_adaptation': self.config.domain_adaptation,
                'knowledge_distillation': self.config.knowledge_distillation
            },
            'transfer_history': self.transfer_history,
            'performance_metrics': self.performance_metrics,
            'available_strategies': list(self.transfer_methods.keys())
        }

# Funciones de utilidad


def create_transfer_learning_config(strategy: str = "feature_extraction",
                                    freeze_layers: int = -1,
                                    source_model_path: str = "") -> TransferLearningConfig:
    """Crea configuración para transfer learning"""
    return TransferLearningConfig(
        transfer_strategy=strategy,
        freeze_layers=freeze_layers,
        source_model_path=source_model_path
    )


def transfer_model_weights(source_model: nn.Module, target_model: nn.Module,
                           freeze_layers: int = -1) -> nn.Module:
    """Transfiere pesos entre modelos compatibles"""

    config = TransferLearningConfig(freeze_layers=freeze_layers)
    manager = TransferLearningManager(config)

    # Crear datos dummy para la transferencia
    dummy_data = torch.randn(1, *getattr(source_model, 'input_shape', (3, 224, 224)))

    return manager.transfer_to_target_task(source_model, target_model, dummy_data)


def compare_transfer_strategies(source_model: nn.Module, target_model: nn.Module,
                                test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, Dict]:
    """Compara diferentes estrategias de transfer learning"""

    strategies = ['feature_extraction', 'domain_adaptation', 'knowledge_distillation']
    results = {}

    for strategy in strategies:
        config = create_transfer_learning_config(strategy=strategy)
        manager = TransferLearningManager(config)

        # Realizar transferencia
        transferred_model = manager.transfer_to_target_task(source_model, target_model, test_data)

        # Evaluar rendimiento
        performance = manager.evaluate_transfer_performance(transferred_model, test_data, test_labels)

        results[strategy] = performance

    return results


# Exportar clases y funciones principales
__all__ = [
    'TransferLearningConfig',
    'BaseTransferLearner',
    'FeatureExtractor',
    'DomainAdapter',
    'KnowledgeDistiller',
    'MultiTaskLearner',
    'TransferLearningManager',
    'create_transfer_learning_config',
    'transfer_model_weights',
    'compare_transfer_strategies'
]

logger.info("RFEN3_RN_7 - Sistema de Aprendizaje Transferido Avanzado cargado correctamente")
