"""
RFEN3_RN_9 - Monitoreo de Rendimiento y Métricas Avanzadas
Implementación de sistema completo de monitoreo y análisis de rendimiento
Incluye: Métricas en tiempo real, visualización, alertas, y análisis de tendencias
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json
import threading
from collections import deque, defaultdict
import warnings
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:
    pass  # dependencia pesada opcional
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del modelo"""
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    weight_norm: float = 0.0
    memory_usage: float = 0.0
    training_time: float = 0.0
    timestamp: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuración para monitoreo de rendimiento"""
    log_frequency: int = 10  # Cada N épocas
    save_frequency: int = 50  # Cada N épocas
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'val_loss_increase': 0.1,
        'gradient_explosion': 10.0,
        'memory_limit': 0.9,
        'accuracy_plateau': 0.001
    })
    visualization_enabled: bool = True
    real_time_monitoring: bool = True
    metrics_to_track: List[str] = field(default_factory=lambda: [
        'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy',
        'learning_rate', 'gradient_norm', 'weight_norm'
    ])
    save_path: str = "./performance_logs"


class BaseMetricCalculator(ABC):
    """Clase base abstracta para calculadores de métricas"""

    @abstractmethod
    def calculate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Calcula métricas específicas"""
        pass


class ClassificationMetrics(BaseMetricCalculator):
    """
    Calculador de métricas para tareas de clasificación
    """

    def calculate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Calcula métricas de clasificación"""

        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)

        # Calcular métricas básicas
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Calcular AUC si es binario
        if len(np.unique(all_targets)) == 2:
            try:
                auc = roc_auc_score(all_targets, all_probabilities[:, 1])
                metrics['auc'] = auc
            except ValueError:
                metrics['auc'] = 0.0

        return metrics


class RegressionMetrics(BaseMetricCalculator):
    """
    Calculador de métricas para tareas de regresión
    """

    def calculate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Calcula métricas de regresión"""

        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)

                all_predictions.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Calcular métricas de regresión
        mse = np.mean((all_targets - all_predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_targets - all_predictions))

        # R² score
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }


class ModelHealthMetrics(BaseMetricCalculator):
    """
    Calculador de métricas de salud del modelo
    """

    def calculate(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Calcula métricas de salud del modelo"""

        metrics = {}

        # Calcular normas de pesos
        weight_norms = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_norms.append(param.data.norm().item())

        metrics['avg_weight_norm'] = np.mean(weight_norms)
        metrics['max_weight_norm'] = np.max(weight_norms)
        metrics['weight_norm_std'] = np.std(weight_norms)

        # Calcular número de parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        metrics['total_parameters'] = total_params
        metrics['trainable_parameters'] = trainable_params
        metrics['parameter_ratio'] = trainable_params / total_params if total_params > 0 else 0

        # Calcular gradientes si están disponibles
        gradient_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad:
                gradient_norms.append(param.grad.data.norm().item())

        if gradient_norms:
            metrics['avg_gradient_norm'] = np.mean(gradient_norms)
            metrics['max_gradient_norm'] = np.max(gradient_norms)
            metrics['gradient_norm_std'] = np.std(gradient_norms)
        else:
            metrics['avg_gradient_norm'] = 0.0
            metrics['max_gradient_norm'] = 0.0
            metrics['gradient_norm_std'] = 0.0

        return metrics


class PerformanceVisualizer:
    """
    Visualizador de métricas de rendimiento
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.figures = {}
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

    def plot_training_curves(self, metrics_history: List[PerformanceMetrics]) -> None:
        """Grafica las curvas de entrenamiento"""

        if not metrics_history:
            return

        epochs = [m.epoch for m in metrics_history]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Métricas de Entrenamiento', fontsize=16)

        # Pérdida
        axes[0, 0].plot(epochs, [m.train_loss for m in metrics_history],
                        label='Train Loss', color=self.colors[0])
        axes[0, 0].plot(epochs, [m.val_loss for m in metrics_history],
                        label='Val Loss', color=self.colors[1])
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Precisión
        axes[0, 1].plot(epochs, [m.train_accuracy for m in metrics_history],
                        label='Train Accuracy', color=self.colors[2])
        axes[0, 1].plot(epochs, [m.val_accuracy for m in metrics_history],
                        label='Val Accuracy', color=self.colors[3])
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning Rate
        axes[1, 0].plot(epochs, [m.learning_rate for m in metrics_history],
                        color=self.colors[4])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)

        # Gradient Norm
        axes[1, 1].plot(epochs, [m.gradient_norm for m in metrics_history],
                        color=self.colors[5])
        axes[1, 1].set_title('Gradient Norm')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.config.save_path}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: List[str] = None) -> None:
        """Grafica matriz de confusión"""

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.tight_layout()
        plt.savefig(f"{self.config.save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, feature_names: List[str],
                                importance_scores: np.ndarray) -> None:
        """Grafica importancia de características"""

        # Ordenar por importancia
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]

        # Tomar top 20
        top_n = min(20, len(sorted_names))
        top_names = sorted_names[:top_n]
        top_scores = sorted_scores[:top_n]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_names)), top_scores)
        plt.yticks(range(len(top_names)), top_names)
        plt.xlabel('Importancia')
        plt.title('Top 20 Características Más Importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{self.config.save_path}/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_learning_rate_schedule(self, metrics_history: List[PerformanceMetrics]) -> None:
        """Grafica el schedule de learning rate"""

        epochs = [m.epoch for m in metrics_history]
        lrs = [m.learning_rate for m in metrics_history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lrs, linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.config.save_path}/learning_rate_schedule.png", dpi=300, bbox_inches='tight')
        plt.close()


class AlertSystem:
    """
    Sistema de alertas para monitoreo de rendimiento
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = []
        self.thresholds = config.alert_thresholds

    def check_alerts(self, current_metrics: PerformanceMetrics,
                     metrics_history: List[PerformanceMetrics]) -> List[Dict]:
        """Verifica condiciones de alerta"""

        alerts = []

        if len(metrics_history) < 2:
            return alerts

        # Alert: Pérdida de validación aumentando
        if 'val_loss_increase' in self.thresholds:
            recent_val_losses = [m.val_loss for m in metrics_history[-5:]]
            if len(recent_val_losses) >= 2:
                loss_increase = recent_val_losses[-1] - recent_val_losses[-2]
                if loss_increase > self.thresholds['val_loss_increase']:
                    alerts.append({
                        'type': 'val_loss_increase',
                        'message': f'Validation loss increased by {loss_increase:.4f}',
                        'severity': 'warning',
                        'timestamp': time.time()
                    })

        # Alert: Explosión de gradientes
        if 'gradient_explosion' in self.thresholds:
            if current_metrics.gradient_norm > self.thresholds['gradient_explosion']:
                alerts.append({
                    'type': 'gradient_explosion',
                    'message': f'Gradient norm is {current_metrics.gradient_norm:.2f}',
                    'severity': 'critical',
                    'timestamp': time.time()
                })

        # Alert: Uso excesivo de memoria
        if 'memory_limit' in self.thresholds:
            if current_metrics.memory_usage > self.thresholds['memory_limit']:
                alerts.append({
                    'type': 'memory_limit',
                    'message': f'Memory usage is {current_metrics.memory_usage:.2%}',
                    'severity': 'warning',
                    'timestamp': time.time()
                })

        # Alert: Plateau de precisión
        if 'accuracy_plateau' in self.thresholds:
            recent_accuracies = [m.val_accuracy for m in metrics_history[-10:]]
            if len(recent_accuracies) >= 5:
                accuracy_std = np.std(recent_accuracies)
                if accuracy_std < self.thresholds['accuracy_plateau']:
                    alerts.append({
                        'type': 'accuracy_plateau',
                        'message': f'Accuracy plateau detected (std: {accuracy_std:.4f})',
                        'severity': 'info',
                        'timestamp': time.time()
                    })

        # Registrar alertas
        for alert in alerts:
            self.alert_history.append(alert)
            logger.warning(f"ALERT: {alert['message']}")

        return alerts

    def get_alert_summary(self) -> Dict:
        """Obtiene resumen de alertas"""

        alert_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for alert in self.alert_history:
            alert_counts[alert['type']] += 1
            severity_counts[alert['severity']] += 1

        return {
            'total_alerts': len(self.alert_history),
            'alert_types': dict(alert_counts),
            'severity_distribution': dict(severity_counts),
            'recent_alerts': self.alert_history[-10:] if self.alert_history else []
        }


class PerformanceAnalyzer:
    """
    Analizador de tendencias y patrones en el rendimiento
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.analysis_cache = {}

    def analyze_trends(self, metrics_history: List[PerformanceMetrics]) -> Dict:
        """Analiza tendencias en las métricas"""

        if len(metrics_history) < 10:
            return {'status': 'insufficient_data'}

        analysis = {}

        # Análisis de pérdida
        train_losses = [m.train_loss for m in metrics_history]
        val_losses = [m.val_loss for m in metrics_history]

        analysis['loss_trends'] = {
            'train_loss_trend': self._calculate_trend(train_losses),
            'val_loss_trend': self._calculate_trend(val_losses),
            'overfitting_risk': self._assess_overfitting(train_losses, val_losses)
        }

        # Análisis de precisión
        train_accuracies = [m.train_accuracy for m in metrics_history]
        val_accuracies = [m.val_accuracy for m in metrics_history]

        analysis['accuracy_trends'] = {
            'train_accuracy_trend': self._calculate_trend(train_accuracies),
            'val_accuracy_trend': self._calculate_trend(val_accuracies),
            'convergence_rate': self._calculate_convergence_rate(val_accuracies)
        }

        # Análisis de estabilidad
        analysis['stability'] = {
            'loss_stability': self._calculate_stability(train_losses),
            'accuracy_stability': self._calculate_stability(val_accuracies),
            'gradient_stability': self._calculate_stability([m.gradient_norm for m in metrics_history])
        }

        return analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula la tendencia de una serie de valores"""

        if len(values) < 3:
            return 'insufficient_data'

        # Usar regresión lineal simple
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)

        if abs(slope) < 0.001:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'

    def _assess_overfitting(self, train_losses: List[float], val_losses: List[float]) -> str:
        """Evalúa el riesgo de sobreajuste"""

        if len(train_losses) < 5 or len(val_losses) < 5:
            return 'insufficient_data'

        # Calcular gap entre train y val
        recent_train = np.mean(train_losses[-5:])
        recent_val = np.mean(val_losses[-5:])
        gap = recent_val - recent_train

        if gap < 0.01:
            return 'low'
        elif gap < 0.05:
            return 'medium'
        else:
            return 'high'

    def _calculate_convergence_rate(self, values: List[float]) -> float:
        """Calcula la tasa de convergencia"""

        if len(values) < 10:
            return 0.0

        # Calcular mejora promedio por época
        improvements = []
        for i in range(1, len(values)):
            improvement = values[i] - values[i-1]
            improvements.append(improvement)

        return np.mean(improvements)

    def _calculate_stability(self, values: List[float]) -> float:
        """Calcula la estabilidad de una métrica"""

        if len(values) < 3:
            return 0.0

        # Usar coeficiente de variación
        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return 0.0

        cv = std_val / abs(mean_val)
        stability = 1.0 / (1.0 + cv)  # Convertir a escala 0-1

        return stability


class PerformanceMonitor:
    """
    Monitor principal de rendimiento
    Coordina todos los componentes de monitoreo
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = []
        self.visualizer = PerformanceVisualizer(config)
        self.alert_system = AlertSystem(config)
        self.analyzer = PerformanceAnalyzer(config)

        # Calculadores de métricas
        self.metric_calculators = {
            'classification': ClassificationMetrics(),
            'regression': RegressionMetrics(),
            'model_health': ModelHealthMetrics()
        }

        # Crear directorio de guardado
        import os
        os.makedirs(config.save_path, exist_ok=True)

        self.monitoring_active = False
        self.monitor_thread = None

    def start_monitoring(self) -> None:
        """Inicia el monitoreo en tiempo real"""

        if self.config.real_time_monitoring and not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Monitoreo en tiempo real iniciado")

    def stop_monitoring(self) -> None:
        """Detiene el monitoreo en tiempo real"""

        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Monitoreo en tiempo real detenido")

    def log_metrics(self, metrics: PerformanceMetrics) -> None:
        """Registra métricas de rendimiento"""

        self.metrics_history.append(metrics)

        # Verificar alertas
        alerts = self.alert_system.check_alerts(metrics, self.metrics_history)

        # Guardar métricas si es necesario
        if len(self.metrics_history) % self.config.save_frequency == 0:
            self._save_metrics()

        # Generar visualizaciones si es necesario
        if self.config.visualization_enabled and len(self.metrics_history) % self.config.log_frequency == 0:
            self._generate_visualizations()

    def calculate_detailed_metrics(self, model: nn.Module,
                                   data_loader: torch.utils.data.DataLoader,
                                   task_type: str = 'classification') -> Dict[str, float]:
        """Calcula métricas detalladas del modelo"""

        metrics = {}

        # Métricas específicas de la tarea
        if task_type in self.metric_calculators:
            task_metrics = self.metric_calculators[task_type].calculate(model, data_loader)
            metrics.update(task_metrics)

        # Métricas de salud del modelo
        health_metrics = self.metric_calculators['model_health'].calculate(model, data_loader)
        metrics.update(health_metrics)

        return metrics

    def _monitoring_loop(self) -> None:
        """Loop de monitoreo en tiempo real"""

        while self.monitoring_active:
            # Aquí se pueden agregar verificaciones adicionales
            time.sleep(1)

    def _save_metrics(self) -> None:
        """Guarda métricas en archivo"""

        metrics_data = []
        for metrics in self.metrics_history:
            metrics_dict = {
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'train_accuracy': metrics.train_accuracy,
                'val_accuracy': metrics.val_accuracy,
                'learning_rate': metrics.learning_rate,
                'gradient_norm': metrics.gradient_norm,
                'weight_norm': metrics.weight_norm,
                'memory_usage': metrics.memory_usage,
                'training_time': metrics.training_time,
                'timestamp': metrics.timestamp,
                'additional_metrics': metrics.additional_metrics
            }
            metrics_data.append(metrics_dict)

        with open(f"{self.config.save_path}/metrics_history.json", 'w') as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"Métricas guardadas: {len(metrics_data)} entradas")

    def _generate_visualizations(self) -> None:
        """Genera visualizaciones de métricas"""

        try:
            self.visualizer.plot_training_curves(self.metrics_history)
            self.visualizer.plot_learning_rate_schedule(self.metrics_history)
            logger.info("Visualizaciones generadas")
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {e}")

    def get_performance_summary(self) -> Dict:
        """Obtiene resumen completo del rendimiento"""

        if not self.metrics_history:
            return {'status': 'no_data'}

        latest_metrics = self.metrics_history[-1]

        # Análisis de tendencias
        trends = self.analyzer.analyze_trends(self.metrics_history)

        # Resumen de alertas
        alert_summary = self.alert_system.get_alert_summary()

        return {
            'current_metrics': {
                'epoch': latest_metrics.epoch,
                'train_loss': latest_metrics.train_loss,
                'val_loss': latest_metrics.val_loss,
                'train_accuracy': latest_metrics.train_accuracy,
                'val_accuracy': latest_metrics.val_accuracy,
                'learning_rate': latest_metrics.learning_rate,
                'gradient_norm': latest_metrics.gradient_norm
            },
            'trends_analysis': trends,
            'alert_summary': alert_summary,
            'total_epochs': len(self.metrics_history),
            'monitoring_active': self.monitoring_active,
            'config': {
                'log_frequency': self.config.log_frequency,
                'save_frequency': self.config.save_frequency,
                'visualization_enabled': self.config.visualization_enabled,
                'real_time_monitoring': self.config.real_time_monitoring
            }
        }

# Funciones de utilidad


def create_performance_monitor(save_path: str = "./performance_logs",
                               log_frequency: int = 10,
                               visualization_enabled: bool = True) -> PerformanceMonitor:
    """Crea un monitor de rendimiento con configuración personalizada"""

    config = MonitoringConfig(
        save_path=save_path,
        log_frequency=log_frequency,
        visualization_enabled=visualization_enabled
    )

    return PerformanceMonitor(config)


def analyze_model_performance(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                              task_type: str = 'classification') -> Dict[str, float]:
    """Analiza el rendimiento de un modelo"""

    config = MonitoringConfig()
    monitor = PerformanceMonitor(config)

    return monitor.calculate_detailed_metrics(model, data_loader, task_type)


def generate_performance_report(metrics_history: List[PerformanceMetrics],
                                save_path: str = "./performance_report") -> Dict:
    """Genera reporte completo de rendimiento"""

    config = MonitoringConfig(save_path=save_path)
    monitor = PerformanceMonitor(config)
    monitor.metrics_history = metrics_history

    # Generar visualizaciones
    monitor._generate_visualizations()

    # Generar análisis
    trends = monitor.analyzer.analyze_trends(metrics_history)

    return {
        'trends_analysis': trends,
        'total_epochs': len(metrics_history),
        'final_metrics': metrics_history[-1] if metrics_history else None,
        'report_generated': True
    }


# Exportar clases y funciones principales
__all__ = [
    'PerformanceMetrics',
    'MonitoringConfig',
    'BaseMetricCalculator',
    'ClassificationMetrics',
    'RegressionMetrics',
    'ModelHealthMetrics',
    'PerformanceVisualizer',
    'AlertSystem',
    'PerformanceAnalyzer',
    'PerformanceMonitor',
    'create_performance_monitor',
    'analyze_model_performance',
    'generate_performance_report'
]

logger.info("RFEN3_RN_9 - Monitoreo de Rendimiento y Métricas Avanzadas cargado correctamente")
