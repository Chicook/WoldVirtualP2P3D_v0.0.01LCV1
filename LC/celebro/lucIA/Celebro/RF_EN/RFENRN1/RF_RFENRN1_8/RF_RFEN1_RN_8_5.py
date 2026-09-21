"""
RF_RFEN1_RN_8_5 - Training Data Processor
==========================================

Neurona especializada en procesamiento de datos de entrenamiento
para refuerzo de pesos en RFEN1_RN_8.

Características:
- Validación de datos de entrenamiento
- Normalización y estandarización
- Data augmentation
- Batch processing
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any


class RFEN1_RN_8_TrainingDataProcessor:
    """
    Neurona procesadora de datos de entrenamiento para RFEN1_RN_8
    Optimiza datos antes de alimentar a la red
    """

    def __init__(self, batch_size: int = 32):
        """
        Inicializar procesador de datos de entrenamiento

        Args:
            batch_size: Tamaño de batch para procesamiento
        """
        self.batch_size = batch_size
        self.processed_batches = []

        self.stats = {
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'total_samples': 0,
            'avg_batch_quality': 0.0
        }

        print(f"[RFEN1_RN_8_5] TrainingDataProcessor inicializado (batch_size={batch_size})")

    def process_batch(self, batch_data: List[np.ndarray], target_network=None) -> Dict[str, Any]:
        """
        Procesar batch de datos de entrenamiento

        Args:
            batch_data: Lista de arrays de datos
            target_network: Red objetivo para refuerzo

        Returns:
            Dict con batch procesado
        """
        self.stats['total_batches'] += 1
        self.stats['total_samples'] += len(batch_data)

        try:
            processed_batch = []
            quality_scores = []

            for sample in batch_data:
                # Validar muestra
                if not self._validate_sample(sample):
                    continue

                # Normalizar muestra
                normalized = self._normalize_sample(sample)

                # Amplificar datos (data augmentation)
                augmented = self._augment_data(normalized)

                # Calcular calidad
                quality = self._assess_sample_quality(augmented)
                quality_scores.append(quality)

                processed_batch.append(augmented)

            if not processed_batch:
                self.stats['failed_batches'] += 1
                return {'success': False, 'reason': 'Empty batch after processing'}

            avg_quality = np.mean(quality_scores)
            self.stats['avg_batch_quality'] = avg_quality
            self.stats['successful_batches'] += 1

            result = {
                'success': True,
                'processed_batch': processed_batch,
                'batch_size': len(processed_batch),
                'avg_quality': avg_quality,
                'quality_scores': quality_scores,
                'timestamp': time.time()
            }

            self.processed_batches.append(result)

            return result

        except Exception as e:
            self.stats['failed_batches'] += 1
            print(f"[RFEN1_RN_8_5] Error procesando batch: {e}")
            return {'success': False, 'error': str(e)}

    def _validate_sample(self, sample: np.ndarray) -> bool:
        """Validar muestra de datos"""
        if sample is None or sample.size == 0:
            return False

        if not isinstance(sample, np.ndarray):
            return False

        if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
            return False

        return True

    def _normalize_sample(self, sample: np.ndarray) -> np.ndarray:
        """Normalizar muestra"""
        if sample.size == 0:
            return sample

        # Normalización estándar
        mean = np.mean(sample)
        std = np.std(sample)

        if std > 0:
            normalized = (sample - mean) / std
        else:
            normalized = sample - mean

        return normalized

    def _augment_data(self, data: np.ndarray) -> np.ndarray:
        """Amplificar datos (data augmentation)"""
        augmented = data.copy()

        # Añadir ruido gaussiano suave
        noise = np.random.normal(0, 0.01, data.shape)
        augmented = augmented + noise

        # Aplicar variaciones leves
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.98, 1.02)
            augmented = augmented * scale

        return augmented

    def _assess_sample_quality(self, sample: np.ndarray) -> float:
        """Evaluar calidad de muestra"""
        if sample.size == 0:
            return 0.0

        variance = np.var(sample)
        range_value = np.ptp(sample)
        mean_value = abs(np.mean(sample))

        quality = min(1.0, (variance + range_value + mean_value) / 3.0)
        return quality

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        return {
            'total_batches': self.stats['total_batches'],
            'successful_batches': self.stats['successful_batches'],
            'failed_batches': self.stats['failed_batches'],
            'total_samples': self.stats['total_samples'],
            'avg_batch_quality': self.stats['avg_batch_quality'],
            'batch_size': self.batch_size
        }
