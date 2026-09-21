"""
ENRN - Entrada de Neuronas para Red Neuronal LucIA
==================================================

Este paquete contiene las neuronas de entrada especializadas para la inteligencia artificial LucIA.
Implementa las mejores prácticas de inicialización de pesos para redes neuronales en 2025.

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025
"""

import logging
import numpy as np
import math
import warnings
from typing import Tuple, Optional, Union, List, Dict, Any
from lucIA.CORE.base import LucIANeuronBase
from lucIA.CORE.initializers import (
    he_initialization as core_he,
    xavier_initialization as core_xavier,
    lecun_initialization as core_lecun
)

# Configuración global para LucIA
LUCIA_CONFIG = {
    'precision': 'float32',
    'random_seed': 42,
    'default_learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout_rate': 0.1,
    'batch_norm_momentum': 0.9,
    'batch_norm_epsilon': 1e-5
}

# Configurar numpy para consistencia
np.random.seed(LUCIA_CONFIG['random_seed'])


# Clase base para todas las neuronas
class NeuronaEntradaBase(LucIANeuronBase):
    """Clase base para todas las neuronas de entrada"""

    def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaEntrada"):
        super().__init__(input_size, output_size, nombre)

    def inicializar_pesos(self) -> None:
        raise NotImplementedError

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = self.obtener_estadisticas_basicas()
        stats.update({
            'input_size': self.input_size,
            'output_size': self.output_size,
            'total_activaciones': len(self.historial_activaciones),
            'total_gradientes': len(self.historial_gradientes)
        })
        return stats

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        return self.obtener_estadisticas()

    def verificar_estabilidad(self) -> Dict[str, bool]:
        return {
            'pesos_inicializados': self.pesos is not None,
            'sesgo_inicializado': self.sesgo is not None
        }

    def resetear_historial(self) -> None:
        self.historial_activaciones = []
        self.historial_gradientes = []

    def recibir_respuesta_llm(self, respuesta: str) -> None:
        """
        Recibe una respuesta de un LLM y reacciona enviando su propio prompt maestro.
        """
        print(f"📥 [Neurona {self.nombre}] Respuesta LLM recibida. Procesando...")
        self.ultima_respuesta_llm = respuesta

        # Preparar prompt maestro propio con lógica de modelos avanzada
        prompt_propio = (
            f"Soy la neurona de entrada {self.nombre}. He recibido este análisis global: {respuesta}. "
            f"Basándote en tu fortaleza específica (lógica, código o eficiencia), "
            f"genera una directiva técnica ultra-específica para el resto de la red neuronal."
        )

        if hasattr(self, 'llm_connector') and self.llm_connector:
            print(f"📤 [Neurona {self.nombre}] Enviando prompt maestro propio a pool de modelos...")

            # Rotación inteligente de modelos según la neurona
            model_to_use = "nemotron_nano_9b"  # Default
            if "ReLU" in self.nombre or "Tanh" in self.nombre:
                model_to_use = "lyria_pro"  # Lógica matemática
            elif "Dropout" in self.nombre:
                model_to_use = "gemma4"  # Rápido y ligero

            respuesta_propia = self.llm_connector.generate_response(prompt_propio, model_key=model_to_use, use_internet=True)
            self.distribuir_a_toda_la_red(respuesta_propia)
        else:
            print(f"⚠️ [Neurona {self.nombre}] No tengo conector LLM para enviar mi prompt.")

    def distribuir_a_toda_la_red(self, directiva: str) -> None:
        """
        Distribuye una directiva a absolutamente todas las neuronas registradas.
        """
        print(f"📢 [Neurona {self.nombre}] Distribuyendo directiva a toda la red: {directiva[:50]}...")
        # ¡Esta función ahora necesita acceso a las instancias de las neuronas, no solo a los pesos!
        # La lógica se moverá al LucIASystem que tiene el mapa de instancias.
        if hasattr(self, '_system_ref'):
            self._system_ref.distribuir_directiva_a_todas_las_neuronas(directiva)
        else:
            print("⚠️ No se puede distribuir la directiva: falta la referencia al sistema principal.")

    def __str__(self) -> str:
        return f"{self.nombre}(entrada={self.input_size}, salida={self.output_size})"

    def __repr__(self) -> str:
        return self.__str__()


# Funciones de utilidad (redireccionadas al CORE)
def he_initialization(fan_in: int, fan_out: int) -> float:
    """Legacy wrapper for CORE He initialization"""
    return core_he((1, 1), fan_in=fan_in).item()


def xavier_initialization(fan_in: int, fan_out: int) -> float:
    """Legacy wrapper for CORE Xavier initialization"""
    return core_xavier((1, 1), fan_in=fan_in, fan_out=fan_out).item()


def lecun_initialization(fan_in: int, fan_out: int) -> float:
    """Legacy wrapper for CORE LeCun initialization"""
    return core_lecun((1, 1), fan_in=fan_in).item()


def normalizar_datos(datos: np.ndarray, metodo: str = 'z_score') -> np.ndarray:
    """Normaliza datos usando el método especificado"""
    if metodo == 'z_score':
        return (datos - np.mean(datos)) / (np.std(datos) + 1e-8)
    elif metodo == 'min_max':
        return (datos - np.min(datos)) / (np.max(datos) - np.min(datos) + 1e-8)
    else:
        return datos


from .EN1_RN import NeuronaEntradaBasica
from .EN2_RN import NeuronaEntradaXavier
from .EN3_RN import NeuronaEntradaLeCun
from .EN4_RN import NeuronaEntradaNormalizada
from .EN5_RN import NeuronaEntradaRegularizada
from .EN6_RN import NeuronaEntradaDropout
from .EN7_RN import NeuronaEntradaBatchNorm
from .EN8_RN import NeuronaEntradaReLU
from .EN9_RN import NeuronaEntradaTanh
from .EN10_RN import NeuronaEntradaSigmoid

__all__ = [
    'NeuronaEntradaBase',
    'NeuronaEntradaBasica',
    'NeuronaEntradaXavier',
    'NeuronaEntradaLeCun',
    'NeuronaEntradaNormalizada',
    'NeuronaEntradaRegularizada',
    'NeuronaEntradaDropout',
    'NeuronaEntradaBatchNorm',
    'NeuronaEntradaReLU',
    'NeuronaEntradaTanh',
    'NeuronaEntradaSigmoid',
    'LUCIA_CONFIG'
]


# Configuración de logging para debugging

# Configurar logging para LucIA
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('ENRN')
logger.info("Paquete ENRN inicializado correctamente para LucIA")
