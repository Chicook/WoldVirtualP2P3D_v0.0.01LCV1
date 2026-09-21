"""
RF_RFENRN1_3_9.py - Gestor de Análisis de Pesos
================================================

Implementa técnicas avanzadas de análisis de pesos para redes neuronales
de aprendizaje por refuerzo. Incluye análisis de distribución, análisis
de importancia, análisis de correlación, análisis de evolución temporal
y técnicas de visualización para comprender el comportamiento de los pesos.

Características:
- Análisis de distribución de pesos por capas
- Análisis de importancia y contribución
- Análisis de correlación entre pesos
- Análisis de evolución temporal
- Detección de patrones y anomalías
- Visualización de análisis
- Reportes automáticos de análisis

Autor: LucIA Development Team
Versión: 3.0.0
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
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger('RFENRN1.RF_RFENRN1_3_9')


@dataclass
class WeightAnalysisConfig:
    """Configuración para análisis de pesos"""
    use_distribution_analysis: bool = True
    use_importance_analysis: bool = True
    use_correlation_analysis: bool = True
    use_temporal_analysis: bool = True
    use_pattern_detection: bool = True
    use_visualization: bool = True
    analysis_frequency: int = 100
    history_length: int = 1000
    correlation_threshold: float = 0.7
    importance_threshold: float = 0.1
    pattern_window: int = 50


class WeightAnalysisManager:
    """
    Gestor de análisis de pesos para redes de refuerzo.

    Implementa técnicas avanzadas de análisis que proporcionan
    insights profundos sobre el comportamiento de los pesos.
    """

    def __init__(self, config: Optional[WeightAnalysisConfig] = None):
        """
        Inicializa el gestor de análisis.

        Args:
            config: Configuración de análisis (opcional)
        """
        self.config = config or WeightAnalysisConfig()
        self.analysis_stats = {
            'total_analyses': 0,
            'distribution_analyses': 0,
            'importance_analyses': 0,
            'correlation_analyses': 0,
            'temporal_analyses': 0,
            'patterns_detected': 0,
            'analysis_quality': 0.0
        }
        self.weight_history = defaultdict(list)
        self.analysis_results = defaultdict(list)
        self.patterns_detected = []

        logger.info("WeightAnalysisManager inicializado")

    def analyze_model_weights(self, model: nn.Module, step: int) -> Dict[str, Any]:
        """
        Analiza pesos del modelo.

        Args:
            model: Modelo PyTorch
            step: Paso actual

        Returns:
            Resultado del análisis
        """
        if step % self.config.analysis_frequency != 0:
            return {'status': 'skipped', 'step': step}

        analysis_result = {
            'step': step,
            'timestamp': time.time(),
            'analyses': {},
            'patterns': [],
            'recommendations': [],
            'quality_score': 0.0
        }

        try:
            # Análisis de distribución
            if self.config.use_distribution_analysis:
                dist_result = self._analyze_distribution(model)
                analysis_result['analyses']['distribution'] = dist_result
                self.analysis_stats['distribution_analyses'] += 1

            # Análisis de importancia
            if self.config.use_importance_analysis:
                importance_result = self._analyze_importance(model)
                analysis_result['analyses']['importance'] = importance_result
                self.analysis_stats['importance_analyses'] += 1

            # Análisis de correlación
            if self.config.use_correlation_analysis:
                corr_result = self._analyze_correlation(model)
                analysis_result['analyses']['correlation'] = corr_result
                self.analysis_stats['correlation_analyses'] += 1

            # Análisis temporal
            if self.config.use_temporal_analysis:
                temporal_result = self._analyze_temporal_evolution(model, step)
                analysis_result['analyses']['temporal'] = temporal_result
                self.analysis_stats['temporal_analyses'] += 1

            # Detección de patrones
            if self.config.use_pattern_detection:
                patterns = self._detect_patterns(model, step)
                analysis_result['patterns'] = patterns
                self.analysis_stats['patterns_detected'] += len(patterns)

            # Generar recomendaciones
            recommendations = self._generate_recommendations(analysis_result)
            analysis_result['recommendations'] = recommendations

            # Calcular calidad del análisis
            quality_score = self._calculate_analysis_quality(analysis_result)
            analysis_result['quality_score'] = quality_score

            # Actualizar estadísticas
            self.analysis_stats['total_analyses'] += 1

            # Registrar resultado
            self.analysis_results['step'].append(analysis_result)
            if len(self.analysis_results['step']) > self.config.history_length:
                self.analysis_results['step'].pop(0)

        except Exception as e:
            logger.error(f"Error en análisis: {e}")
            analysis_result['error'] = str(e)

        return analysis_result

    def _analyze_distribution(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analiza distribución de pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            Resultado del análisis de distribución
        """
        result = {
            'layer_distributions': {},
            'overall_distribution': {},
            'distribution_quality': 0.0
        }

        all_weights = []

        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.data.flatten().cpu().numpy()
                all_weights.extend(weights)

                # Estadísticas por capa
                layer_stats = {
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights),
                    'skewness': self._calculate_skewness(weights),
                    'kurtosis': self._calculate_kurtosis(weights),
                    'zero_ratio': np.sum(weights == 0) / len(weights)
                }

                result['layer_distributions'][name] = layer_stats

                # Actualizar historial
                self.weight_history[name].append(layer_stats)
                if len(self.weight_history[name]) > self.config.history_length:
                    self.weight_history[name].pop(0)

        # Estadísticas generales
        if all_weights:
            result['overall_distribution'] = {
                'mean': np.mean(all_weights),
                'std': np.std(all_weights),
                'min': np.min(all_weights),
                'max': np.max(all_weights),
                'skewness': self._calculate_skewness(all_weights),
                'kurtosis': self._calculate_kurtosis(all_weights),
                'zero_ratio': np.sum(np.array(all_weights) == 0) / len(all_weights)
            }

            # Calcular calidad de distribución
            result['distribution_quality'] = self._calculate_distribution_quality(result)

        return result

    def _analyze_importance(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analiza importancia de pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            Resultado del análisis de importancia
        """
        result = {
            'layer_importance': {},
            'weight_importance': {},
            'importance_ranking': [],
            'importance_score': 0.0
        }

        importance_scores = {}

        for name, param in model.named_parameters():
            if 'weight' in name:
                # Calcular importancia basándose en magnitud y gradientes
                weight_magnitude = torch.norm(param.data).item()

                if param.grad is not None:
                    gradient_magnitude = torch.norm(param.grad).item()
                    importance = weight_magnitude * gradient_magnitude
                else:
                    importance = weight_magnitude

                importance_scores[name] = importance

                # Análisis detallado por peso
                individual_weights = param.data.flatten().cpu().numpy()
                individual_importance = np.abs(individual_weights)

                result['weight_importance'][name] = {
                    'mean_importance': np.mean(individual_importance),
                    'max_importance': np.max(individual_importance),
                    'important_weights_ratio': np.sum(individual_importance >
                                                      np.mean(individual_importance) * 2) / len(individual_importance)
                }

        # Ranking de importancia
        sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        result['importance_ranking'] = sorted_importance

        # Score de importancia general
        if importance_scores:
            result['importance_score'] = np.mean(list(importance_scores.values()))

        return result

    def _analyze_correlation(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analiza correlación entre pesos.

        Args:
            model: Modelo PyTorch

        Returns:
            Resultado del análisis de correlación
        """
        result = {
            'layer_correlations': {},
            'cross_layer_correlations': {},
            'correlation_patterns': [],
            'correlation_score': 0.0
        }

        layer_weights = {}

        # Recopilar pesos por capa
        for name, param in model.named_parameters():
            if 'weight' in name:
                layer_weights[name] = param.data.flatten().cpu().numpy()

        # Correlaciones dentro de capas
        for name, weights in layer_weights.items():
            if len(weights) > 1:
                # Calcular autocorrelación
                autocorr = np.corrcoef(weights[:-1], weights[1:])[0, 1]

                result['layer_correlations'][name] = {
                    'autocorrelation': autocorr,
                    'internal_correlation': np.corrcoef(weights, weights)[0, 1] if len(weights) > 1 else 0
                }

        # Correlaciones entre capas
        layer_names = list(layer_weights.keys())
        for i in range(len(layer_names)):
            for j in range(i + 1, len(layer_names)):
                name1, name2 = layer_names[i], layer_names[j]
                weights1, weights2 = layer_weights[name1], layer_weights[name2]

                # Calcular correlación entre capas
                if len(weights1) == len(weights2):
                    corr = np.corrcoef(weights1, weights2)[0, 1]
                    result['cross_layer_correlations'][f"{name1}_vs_{name2}"] = corr

                    # Detectar patrones de correlación
                    if abs(corr) > self.config.correlation_threshold:
                        result['correlation_patterns'].append({
                            'type': 'high_correlation',
                            'layers': [name1, name2],
                            'correlation': corr
                        })

        # Score de correlación general
        all_correlations = []
        all_correlations.extend([v['autocorrelation'] for v in result['layer_correlations'].values()])
        all_correlations.extend(result['cross_layer_correlations'].values())

        if all_correlations:
            result['correlation_score'] = np.mean(np.abs(all_correlations))

        return result

    def _analyze_temporal_evolution(self, model: nn.Module, step: int) -> Dict[str, Any]:
        """
        Analiza evolución temporal de pesos.

        Args:
            model: Modelo PyTorch
            step: Paso actual

        Returns:
            Resultado del análisis temporal
        """
        result = {
            'evolution_trends': {},
            'stability_metrics': {},
            'convergence_indicators': {},
            'temporal_score': 0.0
        }

        for name, param in model.named_parameters():
            if 'weight' in name and name in self.weight_history:
                history = self.weight_history[name]

                if len(history) >= 3:
                    # Análisis de tendencias
                    means = [h['mean'] for h in history[-10:]]
                    stds = [h['std'] for h in history[-10:]]

                    # Calcular tendencias
                    mean_trend = np.polyfit(range(len(means)), means, 1)[0] if len(means) > 1 else 0
                    std_trend = np.polyfit(range(len(stds)), stds, 1)[0] if len(stds) > 1 else 0

                    result['evolution_trends'][name] = {
                        'mean_trend': mean_trend,
                        'std_trend': std_trend,
                        'trend_stability': 1.0 / (1.0 + abs(mean_trend) + abs(std_trend))
                    }

                    # Métricas de estabilidad
                    mean_stability = 1.0 / (1.0 + np.std(means)) if means else 0
                    std_stability = 1.0 / (1.0 + np.std(stds)) if stds else 0

                    result['stability_metrics'][name] = {
                        'mean_stability': mean_stability,
                        'std_stability': std_stability,
                        'overall_stability': (mean_stability + std_stability) / 2
                    }

                    # Indicadores de convergencia
                    if len(means) >= 5:
                        recent_means = means[-5:]
                        convergence_rate = abs(recent_means[-1] - recent_means[0]) / abs(recent_means[0]) if recent_means[0] != 0 else 0

                        result['convergence_indicators'][name] = {
                            'convergence_rate': convergence_rate,
                            'is_converging': convergence_rate < 0.01
                        }

        # Score temporal general
        if result['stability_metrics']:
            stability_scores = [v['overall_stability'] for v in result['stability_metrics'].values()]
            result['temporal_score'] = np.mean(stability_scores)

        return result

    def _detect_patterns(self, model: nn.Module, step: int) -> List[Dict[str, Any]]:
        """
        Detecta patrones en los pesos.

        Args:
            model: Modelo PyTorch
            step: Paso actual

        Returns:
            Lista de patrones detectados
        """
        patterns = []

        # Detectar patrones de distribución
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.data.flatten().cpu().numpy()

                # Patrón de distribución bimodal
                if self._is_bimodal_distribution(weights):
                    patterns.append({
                        'type': 'bimodal_distribution',
                        'layer': name,
                        'severity': 'medium',
                        'description': 'Distribución bimodal detectada'
                    })

                # Patrón de pesos extremos
                if self._has_extreme_weights(weights):
                    patterns.append({
                        'type': 'extreme_weights',
                        'layer': name,
                        'severity': 'high',
                        'description': 'Pesos extremos detectados'
                    })

                # Patrón de sparsity
                sparsity_ratio = np.sum(weights == 0) / len(weights)
                if sparsity_ratio > 0.5:
                    patterns.append({
                        'type': 'high_sparsity',
                        'layer': name,
                        'severity': 'low',
                        'description': f'Alta sparsity detectada: {sparsity_ratio:.2f}'
                    })

        return patterns

    def _is_bimodal_distribution(self, weights: np.ndarray) -> bool:
        """
        Detecta distribución bimodal.

        Args:
            weights: Array de pesos

        Returns:
            True si es bimodal
        """
        if len(weights) < 100:
            return False

        # Análisis simple de bimodalidad
        hist, bins = np.histogram(weights, bins=20)

        # Buscar dos picos principales
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)

        return len(peaks) >= 2

    def _has_extreme_weights(self, weights: np.ndarray) -> bool:
        """
        Detecta pesos extremos.

        Args:
            weights: Array de pesos

        Returns:
            True si hay pesos extremos
        """
        if len(weights) == 0:
            return False

        mean_weight = np.mean(weights)
        std_weight = np.std(weights)

        if std_weight == 0:
            return False

        # Detectar outliers extremos
        extreme_threshold = 5.0  # 5 desviaciones estándar
        extreme_count = np.sum(np.abs(weights - mean_weight) > extreme_threshold * std_weight)

        return extreme_count > len(weights) * 0.01  # Más del 1% son extremos

    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Genera recomendaciones basadas en el análisis.

        Args:
            analysis_result: Resultado del análisis

        Returns:
            Lista de recomendaciones
        """
        recommendations = []

        # Recomendaciones basadas en distribución
        if 'distribution' in analysis_result['analyses']:
            dist_analysis = analysis_result['analyses']['distribution']
            overall_dist = dist_analysis.get('overall_distribution', {})

            if overall_dist.get('std', 0) > 2.0:
                recommendations.append("Considerar regularización más fuerte - desviación estándar alta")

            if overall_dist.get('zero_ratio', 0) > 0.5:
                recommendations.append("Considerar inicialización más densa - muchos pesos cero")

        # Recomendaciones basadas en patrones
        for pattern in analysis_result.get('patterns', []):
            if pattern['type'] == 'extreme_weights':
                recommendations.append(f"Revisar capa {pattern['layer']} - pesos extremos detectados")
            elif pattern['type'] == 'high_sparsity':
                recommendations.append(f"Considerar pruning en capa {pattern['layer']} - alta sparsity")

        return recommendations

    def _calculate_analysis_quality(self, analysis_result: Dict[str, Any]) -> float:
        """
        Calcula la calidad del análisis.

        Args:
            analysis_result: Resultado del análisis

        Returns:
            Score de calidad (0-1)
        """
        quality_factors = []

        # Factor de completitud
        analyses = analysis_result.get('analyses', {})
        completeness = len(analyses) / 4.0  # 4 tipos de análisis
        quality_factors.append(completeness)

        # Factor de calidad de datos
        if 'distribution' in analyses:
            dist_quality = analyses['distribution'].get('distribution_quality', 0)
            quality_factors.append(dist_quality)

        # Factor de estabilidad
        if 'temporal' in analyses:
            temporal_score = analyses['temporal'].get('temporal_score', 0)
            quality_factors.append(temporal_score)

        return np.mean(quality_factors) if quality_factors else 0.0

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """
        Calcula skewness de los datos.

        Args:
            data: Array de datos

        Returns:
            Skewness
        """
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """
        Calcula kurtosis de los datos.

        Args:
            data: Array de datos

        Returns:
            Kurtosis
        """
        if len(data) < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return 0.0

        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
        return kurtosis

    def _calculate_distribution_quality(self, dist_result: Dict[str, Any]) -> float:
        """
        Calcula calidad de distribución.

        Args:
            dist_result: Resultado de análisis de distribución

        Returns:
            Score de calidad (0-1)
        """
        overall_dist = dist_result.get('overall_distribution', {})

        if not overall_dist:
            return 0.0

        # Factores de calidad
        factors = []

        # Factor de normalidad (skewness y kurtosis cercanos a 0)
        skewness = abs(overall_dist.get('skewness', 0))
        kurtosis = abs(overall_dist.get('kurtosis', 0))
        normality_factor = 1.0 / (1.0 + skewness + kurtosis)
        factors.append(normality_factor)

        # Factor de rango (no demasiado extremo)
        weight_range = overall_dist.get('max', 0) - overall_dist.get('min', 0)
        range_factor = 1.0 / (1.0 + weight_range / 10.0)
        factors.append(range_factor)

        # Factor de densidad (no demasiados ceros)
        zero_ratio = overall_dist.get('zero_ratio', 0)
        density_factor = 1.0 - zero_ratio
        factors.append(density_factor)

        return np.mean(factors)

    def get_analysis_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de análisis.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'analysis_stats': self.analysis_stats.copy(),
            'patterns_detected': self.patterns_detected.copy(),
            'analysis_history_size': len(self.analysis_results.get('step', []))
        }

    def reset_stats(self) -> None:
        """Reinicia las estadísticas de análisis."""
        self.weight_history.clear()
        self.analysis_results.clear()
        self.patterns_detected.clear()
        self.analysis_stats = {
            'total_analyses': 0,
            'distribution_analyses': 0,
            'importance_analyses': 0,
            'correlation_analyses': 0,
            'temporal_analyses': 0,
            'patterns_detected': 0,
            'analysis_quality': 0.0
        }

        logger.info("Estadísticas de análisis reiniciadas")
