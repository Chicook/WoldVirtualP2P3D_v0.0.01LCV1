"""
RF_RFEN1_RN_8_4 - Neural Data Processor
=======================================

Neurona especializada en procesamiento neural de datos de entrenamiento
para RFEN1_RN_8. Implementa extracción de características y pre-procesamiento.

Características:
- Extracción de características avanzada
- Pre-procesamiento de datos
- Normalización adaptativa
- Análisis temporal y espectral
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass


@dataclass
class ProcessedData:
    """Datos procesados neuralmente"""
    features: np.ndarray
    metadata: Dict[str, Any]
    quality_score: float
    processing_time: float
    feature_dimensions: Tuple[int, ...]


class RFEN1_RN_8_NeuralProcessor:
    """
    Neurona procesadora de datos neurales para RFEN1_RN_8
    Procesa datos de entrenamiento y extrae características
    """

    def __init__(self, feature_dim: int = 128):
        """
        Inicializar procesador neural

        Args:
            feature_dim: Dimensión de características extraídas
        """
        self.feature_dim = feature_dim
        self.processed_history = []

        # Filtros de procesamiento
        self.noise_reduction = True
        self.spectral_analysis = True
        self.temporal_analysis = True
        self.normalization = True

        # Estadísticas
        self.stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'avg_quality': 0.0,
            'avg_processing_time': 0.0
        }

        print(f"[RFEN1_RN_8_4] NeuralProcessor inicializado (feature_dim={feature_dim})")

    def process_training_data(self, raw_data: Union[np.ndarray, list], data_type: str = 'voice') -> ProcessedData:
        """
        Procesar datos de entrenamiento

        Args:
            raw_data: Datos crudos
            data_type: Tipo de datos ('voice', 'text', 'mixed')

        Returns:
            ProcessedData con características extraídas
        """
        self.stats['total_processed'] += 1
        start_time = time.time()

        try:
            # Convertir a numpy array
            if isinstance(raw_data, list):
                data = np.array(raw_data)
            else:
                data = raw_data

            # Validar datos
            if data.size == 0:
                raise ValueError("Empty data provided")

            # Procesar según tipo
            if data_type == 'voice':
                processed_features = self._process_voice_data(data)
            elif data_type == 'text':
                processed_features = self._process_text_data(data)
            elif data_type == 'mixed':
                processed_features = self._process_mixed_data(data)
            else:
                processed_features = self._process_generic_data(data)

            # Calcular calidad
            quality_score = self._assess_feature_quality(processed_features)

            # Normalizar si está habilitado
            if self.normalization:
                processed_features = self._normalize_features(processed_features)

            processing_time = time.time() - start_time

            # Actualizar estadísticas
            self.stats['successful_processed'] += 1
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['total_processed'] - 1) + processing_time) /
                self.stats['total_processed']
            )
            self.stats['avg_quality'] = quality_score

            result = ProcessedData(
                features=processed_features,
                metadata={
                    'data_type': data_type,
                    'original_size': data.size,
                    'feature_size': processed_features.size,
                    'compression_ratio': processed_features.size / max(data.size, 1)
                },
                quality_score=quality_score,
                processing_time=processing_time,
                feature_dimensions=processed_features.shape
            )

            self.processed_history.append({
                'timestamp': time.time(),
                'data_type': data_type,
                'quality_score': quality_score,
                'processing_time': processing_time
            })

            return result

        except Exception as e:
            self.stats['failed_processed'] += 1
            print(f"[RFEN1_RN_8_4] Error procesando datos: {e}")

            return ProcessedData(
                features=np.zeros(self.feature_dim),
                metadata={'error': str(e)},
                quality_score=0.0,
                processing_time=time.time() - start_time,
                feature_dimensions=(self.feature_dim,)
            )

    def _process_voice_data(self, data: np.ndarray) -> np.ndarray:
        """Procesar datos de voz"""
        # Extracción de características vocales
        features = np.zeros(self.feature_dim)

        if self.spectral_analysis and len(data) > 0:
            # Análisis espectral
            fft_data = np.fft.fft(data)
            spectral_features = np.abs(fft_data)[:min(self.feature_dim // 2, len(fft_data))]
            features[:len(spectral_features)] = spectral_features / (np.max(np.abs(spectral_features)) + 1e-8)

        if self.temporal_analysis and len(data) > 0:
            # Análisis temporal
            temporal_start = self.feature_dim // 2
            temporal_size = min(self.feature_dim - temporal_start, len(data))
            features[temporal_start:temporal_start + temporal_size] = data[:temporal_size] / (np.max(np.abs(data[:temporal_size])) + 1e-8)

        return features

    def _process_text_data(self, data: np.ndarray) -> np.ndarray:
        """Procesar datos de texto"""
        # Extracción de características textuales
        features = np.zeros(self.feature_dim)

        # Estadísticas básicas del texto
        if len(data) > 0:
            features[0] = np.mean(data)  # Media de caracteres
            features[1] = np.std(data) if len(data) > 1 else 0.0  # Desviación
            features[2] = np.max(data)  # Max
            features[3] = np.min(data)  # Min
            features[4] = len(data) / 1000.0  # Longitud normalizada

            # Características de distribución
            if len(data) > self.feature_dim:
                features[5:min(5 + self.feature_dim, len(data))] = data[:min(self.feature_dim - 5, len(data))]
            else:
                features[5:5 + len(data)] = data

        return features

    def _process_mixed_data(self, data: np.ndarray) -> np.ndarray:
        """Procesar datos mixtos"""
        # Combinar procesamiento de voz y texto
        voice_features = self._process_voice_data(data)
        text_features = self._process_text_data(data)

        # Combinar características
        combined_features = np.zeros(self.feature_dim)
        half_dim = self.feature_dim // 2

        if len(voice_features) > half_dim:
            combined_features[:half_dim] = voice_features[:half_dim]
        else:
            combined_features[:len(voice_features)] = voice_features

        if len(text_features) > half_dim:
            combined_features[half_dim:] = text_features[:half_dim]
        else:
            combined_features[half_dim:half_dim + len(text_features)] = text_features

        return combined_features

    def _process_generic_data(self, data: np.ndarray) -> np.ndarray:
        """Procesar datos genéricos"""
        # Procesamiento genérico
        if data.size == 0:
            return np.zeros(self.feature_dim)

        # Normalizar datos
        normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)

        # Interpolar o truncar según tamaño
        if len(normalized) > self.feature_dim:
            # Downsampling
            indices = np.linspace(0, len(normalized) - 1, self.feature_dim, dtype=int)
            features = normalized[indices]
        elif len(normalized) < self.feature_dim:
            # Padding
            features = np.zeros(self.feature_dim)
            features[:len(normalized)] = normalized
        else:
            features = normalized

        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalizar características"""
        if features.size == 0:
            return features

        # Normalización entre -1 y 1
        normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
        return np.clip(normalized, -1.0, 1.0)

    def _assess_feature_quality(self, features: np.ndarray) -> float:
        """Evaluar calidad de características"""
        if features.size == 0:
            return 0.0

        # Calcular métricas de calidad
        variance_score = np.var(features)
        range_score = np.ptp(features)
        mean_score = abs(np.mean(features))

        # Score combinado
        quality = (variance_score * 0.4 + range_score * 0.3 + mean_score * 0.3)

        return np.clip(quality, 0.0, 1.0)

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del procesador"""
        return {
            'total_processed': self.stats['total_processed'],
            'successful_processed': self.stats['successful_processed'],
            'failed_processed': self.stats['failed_processed'],
            'avg_quality': self.stats['avg_quality'],
            'avg_processing_time': self.stats['avg_processing_time'],
            'feature_dimension': self.feature_dim,
            'processing_history': len(self.processed_history)
        }
