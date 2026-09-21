"""
RF_RFEN1_RN_8_6 - Emotion Analyzer Neural
==========================================

Neurona especializada en análisis de emociones en voz para RFEN1_RN_8.
Detecta y clasifica estados emocionales en síntesis de voz.

Características:
- Detección de emociones en voz
- Clasificación emocional
- Modulación de parámetros de voz
- Refuerzo adaptativo basado en emociones
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class EmotionType(Enum):
    """Tipos de emociones"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"


@dataclass
class EmotionAnalysis:
    """Análisis emocional"""
    detected_emotion: EmotionType
    confidence_score: float
    emotional_features: np.ndarray
    suggested_voice_params: Dict[str, float]
    analysis_timestamp: float


class RFEN1_RN_8_EmotionAnalyzer:
    """
    Neurona analizadora de emociones para RFEN1_RN_8
    Detecta y procesa emociones en síntesis de voz
    """

    def __init__(self):
        """Inicializar analizador de emociones"""
        self.emotion_models = {}
        self.analysis_history = []

        # Parámetros de voz por emoción
        self.emotion_voice_params = {
            EmotionType.HAPPY: {'rate': 170, 'volume': 0.9, 'pitch': 0.8},
            EmotionType.SAD: {'rate': 130, 'volume': 0.7, 'pitch': 0.6},
            EmotionType.ANGRY: {'rate': 180, 'volume': 1.0, 'pitch': 0.9},
            EmotionType.EXCITED: {'rate': 200, 'volume': 0.95, 'pitch': 1.0},
            EmotionType.CALM: {'rate': 140, 'volume': 0.75, 'pitch': 0.7},
            EmotionType.NEUTRAL: {'rate': 150, 'volume': 0.8, 'pitch': 0.75}
        }

        self.stats = {
            'total_analyses': 0,
            'emotion_detections': {e.value: 0 for e in EmotionType},
            'avg_confidence': 0.0
        }

        print("[RFEN1_RN_8_6] EmotionAnalyzer inicializado")

    def analyze_emotion(self, voice_data: np.ndarray, text_context: str = "") -> EmotionAnalysis:
        """
        Analizar emoción en datos de voz

        Args:
            voice_data: Datos de voz
            text_context: Contexto textual

        Returns:
            EmotionAnalysis con resultado
        """
        self.stats['total_analyses'] += 1

        try:
            # Extraer características emocionales
            features = self._extract_emotional_features(voice_data, text_context)

            # Clasificar emoción
            emotion, confidence = self._classify_emotion(features)

            # Obtener parámetros de voz sugeridos
            voice_params = self.emotion_voice_params.get(emotion, self.emotion_voice_params[EmotionType.NEUTRAL])

            result = EmotionAnalysis(
                detected_emotion=emotion,
                confidence_score=confidence,
                emotional_features=features,
                suggested_voice_params=voice_params,
                analysis_timestamp=time.time()
            )

            # Actualizar estadísticas
            self.stats['emotion_detections'][emotion.value] += 1
            self.stats['avg_confidence'] = confidence

            self.analysis_history.append({
                'timestamp': time.time(),
                'emotion': emotion.value,
                'confidence': confidence
            })

            return result

        except Exception as e:
            print(f"[RFEN1_RN_8_6] Error analizando emoción: {e}")

            return EmotionAnalysis(
                detected_emotion=EmotionType.NEUTRAL,
                confidence_score=0.0,
                emotional_features=np.zeros(10),
                suggested_voice_params=self.emotion_voice_params[EmotionType.NEUTRAL],
                analysis_timestamp=time.time()
            )

    def _extract_emotional_features(self, voice_data: np.ndarray, text: str) -> np.ndarray:
        """Extraer características emocionales"""
        features = np.zeros(10)

        if voice_data.size > 0:
            features[0] = np.mean(voice_data)  # Tono promedio
            features[1] = np.std(voice_data)  # Variación de tono
            features[2] = np.ptp(voice_data)  # Rango dinámico
            features[3] = np.max(np.abs(voice_data))  # Intensidad
            features[4] = len(voice_data) / 1000.0  # Duración normalizada

        # Características textuales
        if text:
            text_lower = text.lower()
            features[5] = text.count('!') / max(len(text), 1)  # Exclamaciones
            features[6] = len(text.split()) / 50.0  # Densidad de palabras
            features[7] = 1.0 if any(w in text_lower for w in ['happy', 'joy', 'smile']) else 0.0
            features[8] = 1.0 if any(w in text_lower for w in ['sad', 'sorrow', 'cry']) else 0.0
            features[9] = 1.0 if any(w in text_lower for w in ['angry', 'rage', 'mad']) else 0.0

        return features

    def _classify_emotion(self, features: np.ndarray) -> Tuple[EmotionType, float]:
        """Clasificar emoción"""
        if features.size < 10:
            return EmotionType.NEUTRAL, 0.5

        # Reglas de clasificación simples basadas en características
        intensity = features[3]
        variation = features[1]
        excitement = features[5]

        # Clasificación basada en reglas
        if excitement > 0.1 and intensity > 0.5:
            return EmotionType.EXCITED, 0.8
        elif features[7] > 0.5:
            return EmotionType.HAPPY, 0.7
        elif features[8] > 0.5:
            return EmotionType.SAD, 0.7
        elif features[9] > 0.5 and intensity > 0.7:
            return EmotionType.ANGRY, 0.7
        elif variation < 0.2 and intensity < 0.5:
            return EmotionType.CALM, 0.6
        else:
            return EmotionType.NEUTRAL, 0.5

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas"""
        return {
            'total_analyses': self.stats['total_analyses'],
            'emotion_distribution': self.stats['emotion_detections'],
            'avg_confidence': self.stats['avg_confidence'],
            'supported_emotions': [e.value for e in EmotionType]
        }
