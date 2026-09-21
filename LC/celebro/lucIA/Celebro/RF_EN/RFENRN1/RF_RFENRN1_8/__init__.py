"""
RF_RFENRN1_8 - Sistema de Refuerzo Neural con Síntesis de Voz Realista
========================================================================

Sistema avanzado de neuronas de refuerzo que combina:
- Refuerzo de pesos para RFEN1_RN_8
- Procesamiento optimizado de datos de entrenamiento
- Síntesis de voz realista sin APIs de pago
- Integración con sistema de voz neural de LucIA

Módulos:
- RF_RFEN1_RN_8_1 a RF_RFEN1_RN_8_10: Neuronas especializadas
- Sistema de síntesis de voz con múltiples motores
- Optimización neural de parámetros de voz
- Procesamiento en tiempo real

Autor: WoldVirtual3DlucIA Team
Versión: 0.0.01
"""

__version__ = "0.0.01"
__author__ = "WoldVirtual3DlucIA Team"

# Importar neuronas principales
try:
    from .RF_RFEN1_RN_8_1 import RFEN1_RN_8_VoiceSynthesizer
    from .RF_RFEN1_RN_8_2 import RFEN1_RN_8_VoiceEnhancer
    from .RF_RFEN1_RN_8_3 import RFEN1_RN_8_WeightOptimizer
    from .RF_RFEN1_RN_8_4 import RFEN1_RN_8_NeuralProcessor
    from .RF_RFEN1_RN_8_5 import RFEN1_RN_8_TrainingDataProcessor
    from .RF_RFEN1_RN_8_6 import RFEN1_RN_8_EmotionAnalyzer
    from .RF_RFEN1_RN_8_7 import RFEN1_RN_8_ProsodyController
    from .RF_RFEN1_RN_8_8 import RFEN1_RN_8_SpeechQualityAnalyzer
    from .RF_RFEN1_RN_8_9 import RFEN1_RN_8_MultiVoiceManager
    from .RF_RFEN1_RN_8_10 import RFEN1_RN_8_AdvancedTTS

    NEURONS_AVAILABLE = True
except ImportError as e:
    NEURONS_AVAILABLE = False
    RFEN1_RN_8_VoiceSynthesizer = None
    RFEN1_RN_8_VoiceEnhancer = None
    RFEN1_RN_8_WeightOptimizer = None
    RFEN1_RN_8_NeuralProcessor = None
    RFEN1_RN_8_TrainingDataProcessor = None
    RFEN1_RN_8_EmotionAnalyzer = None
    RFEN1_RN_8_ProsodyController = None
    RFEN1_RN_8_SpeechQualityAnalyzer = None
    RFEN1_RN_8_MultiVoiceManager = None
    RFEN1_RN_8_AdvancedTTS = None
    print(f"[WARNING] Error importando módulos RF_RFENRN1_8: {e}")

# Configuración del módulo
MODULE_CONFIG = {
    'module_name': 'RF_RFENRN1_8',
    'description': 'Sistema de Refuerzo Neural con Síntesis de Voz Realista',
    'version': __version__,
    'components': [
        'VoiceSynthesizer',
        'VoiceEnhancer',
        'WeightOptimizer',
        'NeuralProcessor',
        'TrainingDataProcessor',
        'EmotionAnalyzer',
        'ProsodyController',
        'SpeechQualityAnalyzer',
        'MultiVoiceManager',
        'AdvancedTTS'
    ]
}

__all__ = [
    'RFEN1_RN_8_VoiceSynthesizer',
    'RFEN1_RN_8_VoiceEnhancer',
    'RFEN1_RN_8_WeightOptimizer',
    'RFEN1_RN_8_NeuralProcessor',
    'RFEN1_RN_8_TrainingDataProcessor',
    'RFEN1_RN_8_EmotionAnalyzer',
    'RFEN1_RN_8_ProsodyController',
    'RFEN1_RN_8_SpeechQualityAnalyzer',
    'RFEN1_RN_8_MultiVoiceManager',
    'RFEN1_RN_8_AdvancedTTS',
    'NEURONS_AVAILABLE',
    'MODULE_CONFIG'
]
