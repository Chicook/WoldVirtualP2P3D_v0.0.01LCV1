"""
RF_RFENRN1_4_8.py - Gestor de Análisis de Convergencia y Estabilidad
====================================================================

Implementa técnicas avanzadas de análisis de convergencia y estabilidad para redes
neuronales de aprendizaje por refuerzo. Incluye análisis de gradientes, estabilidad
dinámica y métricas de convergencia.

Características:
- Análisis de convergencia de gradientes
- Detección de gradientes explosivos/vanishing
- Análisis de estabilidad dinámica
- Métricas de convergencia avanzadas
- Detección de oscilaciones
- Análisis de condición de Hessian
- Monitoreo de estabilidad de pesos
- Análisis de espectro de gradientes

Autor: LucIA Development Team
Versión: 4.8.0
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
from dataclasses import dataclass, field
import time
from collections import defaultdict, deque
import matplotlib.pyplot as plt
try:
    from scipy import stats
except ImportError:
    pass  # dependencia pesada opcional

logger = logging.getLogger('RFENRN1.RF_RFENRN1_4_8')


@dataclass
class ConvergenceConfig:
    """Configuración para análisis de convergencia"""
    gradient_norm_threshold: float = 10.0
    convergence_tolerance: float = 1e-6
    stability_window: int = 100
    hessian_analysis: bool = True
    spectral_analysis: bool = True
    oscillation_detection: bool = True

    # Parámetros de análisis
    gradient_history_size: int = 1000
    weight_history_size: int = 500
    loss_history_size: int = 2000

    # Thresholds para detección
    vanishing_gradient_threshold: float = 1e-8
    exploding_gradient_threshold: float = 100.0
    oscillation_threshold: float = 0.1


class GradientAnalyzer:
    """Analizador de gradientes para detección de problemas"""

    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.gradient_history = deque(maxlen=config.gradient_history_size)
        self.gradient_norms = deque(maxlen=config.gradient_history_size)

    def analyze_gradients(self, model: nn.Module) -> Dict[str, Any]:
        """Analiza los gradientes del modelo"""
        analysis = {
            'gradient_norms': {},
            'vanishing_gradients': {},
            'exploding_gradients': {},
            'gradient_distribution': {},
            'overall_health': 'healthy'
        }

        total_norm = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                analysis['gradient_norms'][name] = grad_norm

                # Detectar gradientes vanishing/exploding
                if grad_norm < self.config.vanishing_gradient_threshold:
                    analysis['vanishing_gradients'][name] = grad_norm
                elif grad_norm > self.config.exploding_gradient_threshold:
                    analysis['exploding_gradients'][name] = grad_norm

                # Análisis de distribución
                grad_flat = param.grad.flatten()
                analysis['gradient_distribution'][name] = {
                    'mean': grad_flat.mean().item(),
                    'std': grad_flat.std().item(),
                    'min': grad_flat.min().item(),
                    'max': grad_flat.max().item(),
                    'skewness': self._calculate_skewness(grad_flat),
                    'kurtosis': self._calculate_kurtosis(grad_flat)
                }

                total_norm += grad_norm ** 2
                param_count += 1

        # Norma total de gradientes
        total_norm = math.sqrt(total_norm)
        analysis['total_gradient_norm'] = total_norm

        # Determinar salud general
        if analysis['vanishing_gradients']:
            analysis['overall_health'] = 'vanishing_gradients'
        elif analysis['exploding_gradients']:
            analysis['overall_health'] = 'exploding_gradients'
        elif total_norm > self.config.gradient_norm_threshold:
            analysis['overall_health'] = 'unstable'

        # Guardar en historial
        self.gradient_norms.append(total_norm)

        return analysis

    def _calculate_skewness(self, data: torch.Tensor) -> float:
        """Calcula la skewness de los datos"""
        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0.0
        return ((data - mean) ** 3).mean() / (std ** 3)

    def _calculate_kurtosis(self, data: torch.Tensor) -> float:
        """Calcula la kurtosis de los datos"""
        mean = data.mean()
        std = data.std()
        if std == 0:
            return 0.0
        return ((data - mean) ** 4).mean() / (std ** 4) - 3


class StabilityAnalyzer:
    """Analizador de estabilidad dinámica"""

    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.weight_history = defaultdict(lambda: deque(maxlen=config.weight_history_size))
        self.loss_history = deque(maxlen=config.loss_history_size)

    def analyze_stability(self, model: nn.Module, loss: float) -> Dict[str, Any]:
        """Analiza la estabilidad del modelo"""
        analysis = {
            'weight_stability': {},
            'loss_stability': {},
            'oscillation_detection': {},
            'convergence_rate': 0.0,
            'stability_score': 0.0
        }

        # Análisis de estabilidad de pesos
        for name, param in model.named_parameters():
            current_weight = param.data.clone()
            self.weight_history[name].append(current_weight)

            if len(self.weight_history[name]) > 1:
                weight_changes = []
                for i in range(1, len(self.weight_history[name])):
                    change = (self.weight_history[name][i] - self.weight_history[name][i-1]).norm().item()
                    weight_changes.append(change)

                analysis['weight_stability'][name] = {
                    'mean_change': np.mean(weight_changes),
                    'std_change': np.std(weight_changes),
                    'max_change': np.max(weight_changes),
                    'stability_trend': self._calculate_trend(weight_changes)
                }

        # Análisis de estabilidad de pérdida
        self.loss_history.append(loss)
        if len(self.loss_history) > 1:
            loss_changes = [self.loss_history[i] - self.loss_history[i-1]
                            for i in range(1, len(self.loss_history))]

            analysis['loss_stability'] = {
                'mean_change': np.mean(loss_changes),
                'std_change': np.std(loss_changes),
                'trend': self._calculate_trend(loss_changes),
                'convergence_rate': self._calculate_convergence_rate()
            }

        # Detección de oscilaciones
        analysis['oscillation_detection'] = self._detect_oscillations()

        # Calcular score de estabilidad
        analysis['stability_score'] = self._calculate_stability_score(analysis)

        return analysis

    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula la tendencia de los valores"""
        if len(values) < 2:
            return 'insufficient_data'

        # Regresión lineal simple
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)

        if abs(slope) < 0.001:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'

    def _calculate_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia"""
        if len(self.loss_history) < 10:
            return 0.0

        recent_losses = list(self.loss_history)[-10:]
        convergence_rate = 0.0

        for i in range(1, len(recent_losses)):
            if recent_losses[i-1] != 0:
                rate = abs(recent_losses[i] - recent_losses[i-1]) / abs(recent_losses[i-1])
                convergence_rate += rate

        return convergence_rate / (len(recent_losses) - 1)

    def _detect_oscillations(self) -> Dict[str, Any]:
        """Detecta oscilaciones en el entrenamiento"""
        if len(self.loss_history) < self.config.stability_window:
            return {'detected': False, 'amplitude': 0.0, 'frequency': 0.0}

        recent_losses = list(self.loss_history)[-self.config.stability_window:]

        # Detectar cambios de signo frecuentes
        sign_changes = 0
        for i in range(1, len(recent_losses)):
            if (recent_losses[i] - recent_losses[i-1]) * (recent_losses[i-1] - recent_losses[i-2]) < 0:
                sign_changes += 1

        oscillation_frequency = sign_changes / len(recent_losses)
        oscillation_amplitude = np.std(recent_losses)

        detected = (oscillation_frequency > self.config.oscillation_threshold and
                    oscillation_amplitude > self.config.convergence_tolerance)

        return {
            'detected': detected,
            'frequency': oscillation_frequency,
            'amplitude': oscillation_amplitude,
            'sign_changes': sign_changes
        }

    def _calculate_stability_score(self, analysis: Dict[str, Any]) -> float:
        """Calcula un score de estabilidad general"""
        score = 1.0

        # Penalizar por oscilaciones
        if analysis['oscillation_detection']['detected']:
            score *= 0.5

        # Penalizar por alta variabilidad en pérdida
        if 'loss_stability' in analysis:
            loss_std = analysis['loss_stability'].get('std_change', 0)
            if loss_std > 0.1:
                score *= 0.7

        # Penalizar por cambios grandes en pesos
        for name, stability in analysis['weight_stability'].items():
            if stability['mean_change'] > 0.1:
                score *= 0.8

        return max(0.0, min(1.0, score))


class HessianAnalyzer:
    """Analizador de Hessian para análisis de curvatura"""

    def __init__(self, config: ConvergenceConfig):
        self.config = config

    def analyze_hessian(self, model: nn.Module, loss_fn: callable, data: torch.Tensor) -> Dict[str, Any]:
        """Analiza la matriz Hessian del modelo"""
        if not self.config.hessian_analysis:
            return {'enabled': False}

        analysis = {
            'enabled': True,
            'condition_number': 0.0,
            'eigenvalues': [],
            'curvature_analysis': {},
            'optimization_difficulty': 'unknown'
        }

        try:
            # Calcular gradientes
            loss = loss_fn(model(data))
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Calcular Hessian diagonal aproximado
            hessian_diag = []
            for grad in grads:
                grad_grad = torch.autograd.grad(grad, model.parameters(), grad_outputs=grad, retain_graph=True)
                hessian_diag.append(grad_grad[0].diag())

            # Concatenar diagonal
            hessian_diag = torch.cat([h.flatten() for h in hessian_diag])

            # Análisis de condición
            eigenvals = hessian_diag.abs()
            eigenvals = eigenvals[eigenvals > 1e-8]  # Filtrar valores muy pequeños

            if len(eigenvals) > 0:
                condition_number = eigenvals.max() / eigenvals.min()
                analysis['condition_number'] = condition_number.item()
                analysis['eigenvalues'] = eigenvals.tolist()

                # Determinar dificultad de optimización
                if condition_number > 1000:
                    analysis['optimization_difficulty'] = 'very_difficult'
                elif condition_number > 100:
                    analysis['optimization_difficulty'] = 'difficult'
                elif condition_number > 10:
                    analysis['optimization_difficulty'] = 'moderate'
                else:
                    analysis['optimization_difficulty'] = 'easy'

                # Análisis de curvatura
                analysis['curvature_analysis'] = {
                    'max_curvature': eigenvals.max().item(),
                    'min_curvature': eigenvals.min().item(),
                    'mean_curvature': eigenvals.mean().item(),
                    'curvature_std': eigenvals.std().item()
                }

        except Exception as e:
            logger.warning(f"Error en análisis de Hessian: {e}")
            analysis['error'] = str(e)

        return analysis


class ConvergenceOptimizerManager:
    """
    Gestor de análisis de convergencia y estabilidad para redes de refuerzo.

    Proporciona análisis completo de convergencia, estabilidad y salud del modelo
    con métricas avanzadas y detección automática de problemas.
    """

    def __init__(self, config: Optional[ConvergenceConfig] = None):
        """
        Inicializa el gestor de análisis de convergencia.

        Args:
            config: Configuración de análisis (opcional)
        """
        self.config = config or ConvergenceConfig()
        self.gradient_analyzer = GradientAnalyzer(self.config)
        self.stability_analyzer = StabilityAnalyzer(self.config)
        self.hessian_analyzer = HessianAnalyzer(self.config)

        self.analysis_history = []
        self.metrics = {
            'total_analyses': 0,
            'convergence_detected': 0,
            'stability_issues': 0,
            'gradient_problems': 0,
            'average_stability_score': 0.0
        }

        logger.info("ConvergenceOptimizerManager inicializado")

    def analyze_model(self, model: nn.Module, loss: float,
                      loss_fn: callable = None, data: torch.Tensor = None) -> Dict[str, Any]:
        """
        Realiza análisis completo del modelo.

        Args:
            model: Modelo PyTorch
            loss: Pérdida actual
            loss_fn: Función de pérdida (opcional)
            data: Datos de entrada (opcional)

        Returns:
            Análisis completo del modelo
        """
        start_time = time.time()

        # Análisis de gradientes
        gradient_analysis = self.gradient_analyzer.analyze_gradients(model)

        # Análisis de estabilidad
        stability_analysis = self.stability_analyzer.analyze_stability(model, loss)

        # Análisis de Hessian (si está disponible)
        hessian_analysis = {}
        if loss_fn is not None and data is not None:
            hessian_analysis = self.hessian_analyzer.analyze_hessian(model, loss_fn, data)

        # Análisis combinado
        combined_analysis = {
            'timestamp': time.time(),
            'gradient_analysis': gradient_analysis,
            'stability_analysis': stability_analysis,
            'hessian_analysis': hessian_analysis,
            'overall_health': self._assess_overall_health(gradient_analysis, stability_analysis),
            'recommendations': self._generate_recommendations(gradient_analysis, stability_analysis, hessian_analysis),
            'analysis_time': time.time() - start_time
        }

        # Actualizar métricas
        self._update_metrics(combined_analysis)

        # Guardar en historial
        self.analysis_history.append(combined_analysis)

        logger.info(f"Análisis completado en {combined_analysis['analysis_time']:.3f}s")

        return combined_analysis

    def _assess_overall_health(self, gradient_analysis: Dict, stability_analysis: Dict) -> str:
        """Evalúa la salud general del modelo"""
        health_score = 1.0

        # Evaluar salud de gradientes
        if gradient_analysis['overall_health'] != 'healthy':
            health_score *= 0.5

        # Evaluar estabilidad
        stability_score = stability_analysis.get('stability_score', 1.0)
        health_score *= stability_score

        # Evaluar oscilaciones
        if stability_analysis['oscillation_detection']['detected']:
            health_score *= 0.7

        if health_score > 0.8:
            return 'excellent'
        elif health_score > 0.6:
            return 'good'
        elif health_score > 0.4:
            return 'fair'
        else:
            return 'poor'

    def _generate_recommendations(self, gradient_analysis: Dict, stability_analysis: Dict,
                                  hessian_analysis: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []

        # Recomendaciones de gradientes
        if gradient_analysis['overall_health'] == 'vanishing_gradients':
            recommendations.append("Considerar técnicas para evitar gradientes vanishing (batch norm, skip connections)")
        elif gradient_analysis['overall_health'] == 'exploding_gradients':
            recommendations.append("Aplicar gradient clipping para controlar gradientes explosivos")

        # Recomendaciones de estabilidad
        if stability_analysis['oscillation_detection']['detected']:
            recommendations.append("Reducir learning rate o usar scheduler para estabilizar entrenamiento")

        if stability_analysis.get('stability_score', 1.0) < 0.5:
            recommendations.append("Considerar técnicas de regularización para mejorar estabilidad")

        # Recomendaciones de Hessian
        if hessian_analysis.get('enabled', False):
            difficulty = hessian_analysis.get('optimization_difficulty', 'unknown')
            if difficulty in ['difficult', 'very_difficult']:
                recommendations.append("Usar optimizadores de segunda orden o técnicas de preconditioning")

        return recommendations

    def _update_metrics(self, analysis: Dict[str, Any]):
        """Actualiza las métricas del análisis"""
        self.metrics['total_analyses'] += 1

        # Detectar convergencia
        if analysis['stability_analysis'].get('convergence_rate', 1.0) < self.config.convergence_tolerance:
            self.metrics['convergence_detected'] += 1

        # Detectar problemas de estabilidad
        if analysis['stability_analysis'].get('stability_score', 1.0) < 0.5:
            self.metrics['stability_issues'] += 1

        # Detectar problemas de gradientes
        if analysis['gradient_analysis']['overall_health'] != 'healthy':
            self.metrics['gradient_problems'] += 1

        # Actualizar score promedio de estabilidad
        current_score = analysis['stability_analysis'].get('stability_score', 0.0)
        self.metrics['average_stability_score'] = (
            (self.metrics['average_stability_score'] * (self.metrics['total_analyses'] - 1) + current_score) /
            self.metrics['total_analyses']
        )

    def get_convergence_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del análisis de convergencia"""
        return {
            'config': self.config,
            'metrics': self.metrics.copy(),
            'total_analyses': self.metrics['total_analyses'],
            'convergence_rate': self.metrics['convergence_detected'] / max(1, self.metrics['total_analyses']),
            'stability_issues_rate': self.metrics['stability_issues'] / max(1, self.metrics['total_analyses']),
            'gradient_problems_rate': self.metrics['gradient_problems'] / max(1, self.metrics['total_analyses']),
            'average_stability_score': self.metrics['average_stability_score'],
            'history_length': len(self.analysis_history)
        }

    def plot_convergence_analysis(self, save_path: str = None) -> None:
        """Genera gráficos de análisis de convergencia"""
        if not self.analysis_history:
            logger.warning("No hay historial de análisis para graficar")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extraer datos del historial
        timestamps = [a['timestamp'] for a in self.analysis_history]
        gradient_norms = [a['gradient_analysis']['total_gradient_norm'] for a in self.analysis_history]
        stability_scores = [a['stability_analysis']['stability_score'] for a in self.analysis_history]
        convergence_rates = [a['stability_analysis']['loss_stability'].get('convergence_rate', 0)
                             for a in self.analysis_history]

        # Gráfico de normas de gradientes
        axes[0, 0].plot(timestamps, gradient_norms)
        axes[0, 0].set_title('Gradient Norms Over Time')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].grid(True)

        # Gráfico de scores de estabilidad
        axes[0, 1].plot(timestamps, stability_scores)
        axes[0, 1].set_title('Stability Scores Over Time')
        axes[0, 1].set_ylabel('Stability Score')
        axes[0, 1].grid(True)

        # Gráfico de tasas de convergencia
        axes[1, 0].plot(timestamps, convergence_rates)
        axes[1, 0].set_title('Convergence Rates Over Time')
        axes[1, 0].set_ylabel('Convergence Rate')
        axes[1, 0].grid(True)

        # Gráfico de salud general
        health_scores = []
        for a in self.analysis_history:
            health = a['overall_health']
            score = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.4}.get(health, 0.0)
            health_scores.append(score)

        axes[1, 1].plot(timestamps, health_scores)
        axes[1, 1].set_title('Overall Health Over Time')
        axes[1, 1].set_ylabel('Health Score')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Gráficos guardados en {save_path}")
        else:
            plt.show()

    def save_state(self, path: str) -> None:
        """Guarda el estado del analizador"""
        torch.save({
            'config': self.config,
            'metrics': self.metrics,
            'analysis_history': self.analysis_history,
            'gradient_history': list(self.gradient_analyzer.gradient_norms),
            'weight_history': dict(self.stability_analyzer.weight_history),
            'loss_history': list(self.stability_analyzer.loss_history)
        }, path)
        logger.info(f"Estado del analizador guardado en {path}")

    def load_state(self, path: str) -> None:
        """Carga el estado del analizador"""
        checkpoint = torch.load(path)

        self.config = checkpoint.get('config', self.config)
        self.metrics = checkpoint.get('metrics', self.metrics)
        self.analysis_history = checkpoint.get('analysis_history', [])

        # Restaurar historiales
        gradient_norms = checkpoint.get('gradient_history', [])
        self.gradient_analyzer.gradient_norms = deque(gradient_norms, maxlen=self.config.gradient_history_size)

        weight_history = checkpoint.get('weight_history', {})
        for name, history in weight_history.items():
            self.stability_analyzer.weight_history[name] = deque(history, maxlen=self.config.weight_history_size)

        loss_history = checkpoint.get('loss_history', [])
        self.stability_analyzer.loss_history = deque(loss_history, maxlen=self.config.loss_history_size)

        logger.info(f"Estado del analizador cargado desde {path}")

    def reset_state(self) -> None:
        """Reinicia el estado del analizador"""
        self.metrics = {
            'total_analyses': 0,
            'convergence_detected': 0,
            'stability_issues': 0,
            'gradient_problems': 0,
            'average_stability_score': 0.0
        }

        self.analysis_history.clear()
        self.gradient_analyzer.gradient_norms.clear()
        self.stability_analyzer.weight_history.clear()
        self.stability_analyzer.loss_history.clear()

        logger.info("Estado del analizador reiniciado")
