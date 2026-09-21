"""
RF_RFEN1_RN_8_8 - Speech Quality Analyzer Neural
==================================================

Neurona especializada en análisis de calidad de síntesis de voz para RFEN1_RN_8.
Evalúa y optimiza calidad en tiempo real.

Características:
- Análisis de calidad en tiempo real
- Métricas de claridad y naturalidad
- Detección de artefactos
- Optimización adaptativa
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Métricas de calidad de voz"""
    clarity_score: float  # 0.0 a 1.0
    naturalness_score: float  # 0.0 a 1.0
    stability_score: float  # 0.0 a 1.0
    artifact_score: float  # 0.0 a 1.0 (menos es mejor)
    overall_quality: float  # 0.0 a 1.0


class RFEN1_RN_8_SpeechQualityAnalyzer:
    """
    Neurona analizadora de calidad de voz para RFEN1_RN_8
    Evalúa y mejora calidad de síntesis
    """

    def __init__(self):
        """Inicializar analizador de calidad"""
        self.quality_history = []
        self.optimization_history = []

        # Thresholds de calidad
        self.quality_threshold = 0.7
        self.clarity_threshold = 0.6
        self.naturalness_threshold = 0.65

        self.stats = {
            'total_analyses': 0,
            'high_quality_count': 0,
            'medium_quality_count': 0,
            'low_quality_count': 0,
            'avg_overall_quality': 0.0
        }

        print("[RFEN1_RN_8_8] SpeechQualityAnalyzer inicializado")

    def analyze_quality(self, voice_data: np.ndarray, target_network=None) -> QualityMetrics:
        """
        Analizar calidad de síntesis de voz

        Args:
            voice_data: Datos de voz a analizar
            target_network: Red objetivo

        Returns:
            QualityMetrics con métricas de calidad
        """
        self.stats['total_analyses'] += 1

        try:
            # Calcular métricas individuales
            clarity = self._calculate_clarity(voice_data)
            naturalness = self._calculate_naturalness(voice_data)
            stability = self._calculate_stability(voice_data)
            artifacts = self._detect_artifacts(voice_data)

            # Calcular calidad overall
            overall = (clarity * 0.3 + naturalness * 0.3 + stability * 0.2 + (1.0 - artifacts) * 0.2)

            metrics = QualityMetrics(
                clarity_score=clarity,
                naturalness_score=naturalness,
                stability_score=stability,
                artifact_score=artifacts,
                overall_quality=overall
            )

            # Actualizar estadísticas
            if overall >= 0.8:
                self.stats['high_quality_count'] += 1
            elif overall >= 0.5:
                self.stats['medium_quality_count'] += 1
            else:
                self.stats['low_quality_count'] += 1

            self.stats['avg_overall_quality'] = (
                (self.stats['avg_overall_quality'] * (self.stats['total_analyses'] - 1) + overall) /
                self.stats['total_analyses']
            )

            # Guardar en historial
            analysis_info = {
                'timestamp': time.time(),
                'clarity': clarity,
                'naturalness': naturalness,
                'stability': stability,
                'artifacts': artifacts,
                'overall': overall
            }
            self.quality_history.append(analysis_info)

            # Optimizar si la calidad es baja
            if overall < self.quality_threshold:
                self._optimize_for_quality(voice_data, target_network)

            return metrics

        except Exception as e:
            print(f"[RFEN1_RN_8_8] Error analizando calidad: {e}")

            return QualityMetrics(
                clarity_score=0.0,
                naturalness_score=0.0,
                stability_score=0.0,
                artifact_score=1.0,
                overall_quality=0.0
            )

    def _calculate_clarity(self, voice_data: np.ndarray) -> float:
        """Calcular claridad de voz"""
        if voice_data.size == 0:
            return 0.0

        # SNR aproximada
        signal_power = np.mean(voice_data ** 2)
        noise_estimate = np.std(voice_data) * 0.1

        snr = signal_power / (noise_estimate + 1e-8)
        clarity = min(1.0, snr / 20.0)

        return clarity

    def _calculate_naturalness(self, voice_data: np.ndarray) -> float:
        """Calcular naturalidad de voz"""
        if voice_data.size == 0:
            return 0.0

        # Suavidad temporal
        temporal_smoothness = 1.0 - min(1.0, np.std(np.diff(voice_data)))

        # Continuidad espectral
        if voice_data.size > 10:
            fft_data = np.fft.fft(voice_data)
            spectral_continuity = 1.0 - min(1.0, np.std(np.abs(fft_data)) / np.mean(np.abs(fft_data)))
        else:
            spectral_continuity = 0.5

        naturalness = (temporal_smoothness * 0.6 + spectral_continuity * 0.4)

        return naturalness

    def _calculate_stability(self, voice_data: np.ndarray) -> float:
        """Calcular estabilidad de voz"""
        if voice_data.size < 2:
            return 0.5

        # Variación temporal
        temporal_var = np.std(np.abs(np.diff(voice_data)))

        # Variación de amplitud
        amplitude_var = np.std(voice_data)

        # Score de estabilidad (menos variación = más estable)
        stability = 1.0 - min(1.0, (temporal_var + amplitude_var) / 2.0)

        return stability

    def _detect_artifacts(self, voice_data: np.ndarray) -> float:
        """Detectar artefactos en voz"""
        if voice_data.size == 0:
            return 1.0

        # Detectar clipping
        clip_threshold = 0.95
        clip_count = np.sum(np.abs(voice_data) > clip_threshold) / len(voice_data)

        # Detectar discontinuidades
        if voice_data.size > 1:
            diff = np.abs(np.diff(voice_data))
            discontinuity_count = np.sum(diff > np.mean(diff) * 3) / len(diff)
        else:
            discontinuity_count = 0.0

        artifact_score = (clip_count * 0.5 + discontinuity_count * 0.5)

        return min(1.0, artifact_score)

    def _optimize_for_quality(self, voice_data: np.ndarray, target_network):
        """Optimizar para mejorar calidad"""
        try:
            # Detectar problemas específicos
            clarity = self._calculate_clarity(voice_data)
            naturalness = self._calculate_naturalness(voice_data)
            stability = self._calculate_stability(voice_data)

            optimization_suggestions = []

            if clarity < self.clarity_threshold:
                optimization_suggestions.append('improve_clarity')

            if naturalness < self.naturalness_threshold:
                optimization_suggestions.append('improve_naturalness')

            if stability < 0.6:
                optimization_suggestions.append('improve_stability')

            # Guardar optimizaciones
            opt_info = {
                'timestamp': time.time(),
                'suggestions': optimization_suggestions,
                'scores': {
                    'clarity': clarity,
                    'naturalness': naturalness,
                    'stability': stability
                }
            }
            self.optimization_history.append(opt_info)

        except Exception as e:
            print(f"[RFEN1_RN_8_8] Error optimizando calidad: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        return {
            'total_analyses': self.stats['total_analyses'],
            'high_quality_count': self.stats['high_quality_count'],
            'medium_quality_count': self.stats['medium_quality_count'],
            'low_quality_count': self.stats['low_quality_count'],
            'avg_overall_quality': self.stats['avg_overall_quality'],
            'quality_distribution': {
                'high': self.stats['high_quality_count'],
                'medium': self.stats['medium_quality_count'],
                'low': self.stats['low_quality_count']
            },
            'history_length': len(self.quality_history),
            'optimizations': len(self.optimization_history)
        }
