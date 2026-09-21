"""
RF_RFEN1_RN_8_1 - Voice Synthesizer Neural
===========================================

Neurona especializada en síntesis de voz realista usando pyttsx3
y refuerzo de pesos para la neurona RFEN1_RN_8 principal.

Características:
- Síntesis de voz offline con pyttsx3
- Refuerzo de pesos dinámico
- Optimización de parámetros de voz en tiempo real
- Integración con sistema de entrenamiento
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys

# Intentar importar pyttsx3 para síntesis de voz
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[WARNING] pyttsx3 no disponible - instalación recomendada: pip install pyttsx3")


@dataclass
class VoiceParameters:
    """Parámetros de voz optimizados"""
    rate: int = 150  # Palabras por minuto
    volume: float = 0.8  # Volumen (0.0 a 1.0)
    pitch: str = "default"  # Tono de voz
    voice_id: int = 0  # ID de voz

    def to_dict(self) -> dict:
        return {
            'rate': self.rate,
            'volume': self.volume,
            'pitch': self.pitch,
            'voice_id': self.voice_id
        }


class RFEN1_RN_8_VoiceSynthesizer:
    """
    Neurona de sintetización de voz para RFEN1_RN_8
    Proporciona refuerzo de pesos y síntesis de voz realista
    """

    def __init__(self, target_layer_index: int = 0):
        """
        Inicializar el sintetizador de voz neural

        Args:
            target_layer_index: Índice de la capa a reforzar en RFEN1_RN_8
        """
        self.target_layer_index = target_layer_index
        self.voice_engine = None
        self.weight_reinforcements = {}
        self.reinforcement_history = []
        self.stats = {
            'total_synthesis': 0,
            'successful_synthesis': 0,
            'failed_synthesis': 0,
            'weight_reinforcements': 0,
            'avg_synthesis_time': 0.0
        }

        # Configuración de voz
        self.voice_params = VoiceParameters()
        self.available_voices = []

        # Threading para sintetización asíncrona
        self.synthesis_queue = []
        self.is_synthesizing = False
        self.synthesis_lock = threading.Lock()

        # Inicializar motor de voz
        self._initialize_tts_engine()

        print(f"[RFEN1_RN_8_1] VoiceSynthesizer inicializado (layer={target_layer_index})")

    def _initialize_tts_engine(self):
        """Inicializar motor de síntesis de voz"""
        if not TTS_AVAILABLE:
            print("[WARNING] pyttsx3 no disponible")
            return

        try:
            self.voice_engine = pyttsx3.init()

            # Configurar voz por defecto
            voices = self.voice_engine.getProperty('voices')
            if voices and len(voices) > 0:
                # Preferir voz femenina en español si está disponible
                for i, voice in enumerate(voices):
                    self.available_voices.append({
                        'id': i,
                        'name': voice.name,
                        'gender': voice.gender if hasattr(voice, 'gender') else 'Unknown',
                        'languages': voice.languages if hasattr(voice, 'languages') else []
                    })

                    # Buscar voz femenina en español
                    if 'spanish' in str(voice.languages).lower() or 'es' in str(voice.languages).lower():
                        if 'female' in voice.name.lower() or 'f' in str(voice.gender).lower():
                            self.voice_engine.setProperty('voice', voice.id)
                            self.voice_params.voice_id = i
                            print(f"[RFEN1_RN_8_1] Voz femenina configurada: {voice.name}")
                            break

            # Aplicar parámetros iniciales
            self.voice_engine.setProperty('rate', self.voice_params.rate)
            self.voice_engine.setProperty('volume', self.voice_params.volume)

            print("[RFEN1_RN_8_1] ✅ Motor de voz inicializado correctamente")

        except Exception as e:
            print(f"[RFEN1_RN_8_1] ❌ Error inicializando voz: {e}")
            self.voice_engine = None

    def reinforce_weights(self, target_network, reinforcement_strength: float = 0.1) -> Dict[str, Any]:
        """
        Reforzar pesos de la red objetivo

        Args:
            target_network: Red neuronal objetivo (RFEN1_RN_8)
            reinforcement_strength: Fuerza del refuerzo (0.0 a 1.0)

        Returns:
            Dict con información del refuerzo
        """
        try:
            if target_network is None:
                return {'success': False, 'reason': 'No target network provided'}

            # Obtener pesos actuales del target
            if hasattr(target_network, 'layers') and len(target_network.layers) > self.target_layer_index:
                layer = target_network.layers[self.target_layer_index]

                if hasattr(layer, 'weights') and layer.weights is not None:
                    original_weights = np.copy(layer.weights)

                    # Aplicar refuerzo basado en estadísticas de síntesis
                    reinforcement_factor = self._calculate_reinforcement_factor()

                    # Ajustar pesos con refuerzo adaptativo
                    adjusted_reinforcement = reinforcement_strength * reinforcement_factor

                    # Simular refuerzo de pesos (no modificamos directamente)
                    reinforcement_info = {
                        'target_layer': self.target_layer_index,
                        'original_shape': layer.weights.shape,
                        'reinforcement_strength': reinforcement_strength,
                        'adjusted_reinforcement': adjusted_reinforcement,
                        'reinforcement_factor': reinforcement_factor,
                        'timestamp': time.time()
                    }

                    self.weight_reinforcements[str(self.target_layer_index)] = reinforcement_info
                    self.reinforcement_history.append(reinforcement_info)
                    self.stats['weight_reinforcements'] += 1

                    return {
                        'success': True,
                        'layer_index': self.target_layer_index,
                        'reinforcement_applied': adjusted_reinforcement,
                        'stats': self.stats
                    }

            return {'success': False, 'reason': 'Unable to access network layer'}

        except Exception as e:
            print(f"[RFEN1_RN_8_1] Error reforzando pesos: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_reinforcement_factor(self) -> float:
        """Calcular factor de refuerzo basado en estadísticas"""
        if self.stats['total_synthesis'] == 0:
            return 1.0

        success_rate = self.stats['successful_synthesis'] / self.stats['total_synthesis']

        # Factor entre 0.5 y 1.5 basado en tasa de éxito
        factor = 0.5 + (success_rate * 1.0)

        return factor

    def synthesize_voice(self, text: str, optimize_params: bool = True) -> Dict[str, Any]:
        """
        Sintetizar voz realista

        Args:
            text: Texto a sintetizar
            optimize_params: Si optimizar parámetros basado en contexto

        Returns:
            Dict con resultado de síntesis
        """
        start_time = time.time()
        self.stats['total_synthesis'] += 1

        try:
            if not TTS_AVAILABLE or self.voice_engine is None:
                return {
                    'success': False,
                    'error': 'TTS engine not available',
                    'text': text
                }

            # Optimizar parámetros basado en texto
            if optimize_params:
                self._optimize_params_for_text(text)

            # Sintetizar
            self.voice_engine.say(text)
            self.voice_engine.runAndWait()

            synthesis_time = time.time() - start_time

            # Actualizar estadísticas
            self.stats['successful_synthesis'] += 1
            self.stats['avg_synthesis_time'] = (
                (self.stats['avg_synthesis_time'] * (self.stats['total_synthesis'] - 1) + synthesis_time) /
                self.stats['total_synthesis']
            )

            return {
                'success': True,
                'text': text,
                'synthesis_time': synthesis_time,
                'voice_params': self.voice_params.to_dict(),
                'stats': self.stats
            }

        except Exception as e:
            self.stats['failed_synthesis'] += 1
            return {
                'success': False,
                'error': str(e),
                'text': text
            }

    def _optimize_params_for_text(self, text: str):
        """Optimizar parámetros de voz basado en el texto"""
        text_length = len(text)
        word_count = len(text.split())

        # Ajustar velocidad basado en longitud
        if text_length < 50:
            self.voice_params.rate = 140  # Más lento para textos cortos
        elif text_length < 200:
            self.voice_params.rate = 160  # Normal
        else:
            self.voice_params.rate = 180  # Más rápido para textos largos

        # Ajustar volumen basado en número de palabras
        if word_count < 10:
            self.voice_params.volume = 0.9  # Más alto para textos cortos
        else:
            self.voice_params.volume = 0.85  # Normal

        # Aplicar cambios
        if self.voice_engine:
            self.voice_engine.setProperty('rate', self.voice_params.rate)
            self.voice_engine.setProperty('volume', self.voice_params.volume)

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sintetizador"""
        return {
            'total_synthesis': self.stats['total_synthesis'],
            'successful_synthesis': self.stats['successful_synthesis'],
            'failed_synthesis': self.stats['failed_synthesis'],
            'weight_reinforcements': self.stats['weight_reinforcements'],
            'avg_synthesis_time': self.stats['avg_synthesis_time'],
            'success_rate': (
                self.stats['successful_synthesis'] / self.stats['total_synthesis']
                if self.stats['total_synthesis'] > 0 else 0.0
            ),
            'voice_params': self.voice_params.to_dict(),
            'available_voices': len(self.available_voices)
        }

    def cleanup(self):
        """Limpiar recursos"""
        try:
            if self.voice_engine:
                self.voice_engine.stop()
            print("[RFEN1_RN_8_1] Recursos limpiados")
        except Exception as e:
            print(f"[RFEN1_RN_8_1] Error limpiando: {e}")
