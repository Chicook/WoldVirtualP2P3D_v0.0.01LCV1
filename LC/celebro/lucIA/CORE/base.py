"""
lucIA.CORE.base - Clases base para el sistema neuronal de LucIA
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class LucIANeuronBase:
    """Clase base para todas las neuronas del sistema LucIA."""

    def __init__(self, input_size: int, output_size: int, nombre: str = "LucIANeuron"):
        self.input_size = input_size
        self.output_size = output_size
        self.nombre = nombre
        self.pesos: Optional[np.ndarray] = None
        self.sesgo: Optional[np.ndarray] = None
        self.historial_activaciones: List[np.ndarray] = []
        self.historial_gradientes: List[Dict[str, Any]] = []
        self.ultima_respuesta_llm: Optional[str] = None
        self.llm_connector = None
        self._system_ref = None

    def inicializar_pesos(self) -> None:
        """Inicialización por defecto"""
        std = float(np.sqrt(2.0 / (self.input_size + self.output_size)))
        self.pesos = np.random.randn(self.input_size, self.output_size).astype(np.float32) * std
        self.sesgo = np.zeros((1, self.output_size), dtype=np.float32)

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None:
            self.inicializar_pesos()
        salida = np.dot(entrada, self.pesos) + self.sesgo
        self.historial_activaciones.append(salida.copy())
        return salida

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({
            'pesos': grad_pesos.copy(),
            'norma': float(np.linalg.norm(grad_pesos))
        })
        return grad_pesos, grad_sesgo

    def obtener_estadisticas_basicas(self) -> Dict[str, Any]:
        stats = {
            'nombre': self.nombre,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'tiene_pesos': self.pesos is not None,
            'tiene_sesgo': self.sesgo is not None,
        }
        if self.pesos is not None:
            stats['pesos_mean'] = float(np.mean(self.pesos))
            stats['pesos_std'] = float(np.std(self.pesos))
            stats['pesos_shape'] = list(self.pesos.shape)
        return stats

    def exportar_pesos(self) -> Dict[str, Any]:
        """Exporta los pesos y sesgos a un diccionario serializable."""
        return {
            'nombre': self.nombre,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'pesos': self.pesos.tolist() if self.pesos is not None else None,
            'sesgo': self.sesgo.tolist() if self.sesgo is not None else None
        }

    def cargar_pesos(self, datos: Dict[str, Any]) -> None:
        """Carga pesos y sesgos desde un diccionario."""
        if 'pesos' in datos and datos['pesos'] is not None:
            self.pesos = np.array(datos['pesos'], dtype=np.float32)
        if 'sesgo' in datos and datos['sesgo'] is not None:
            self.sesgo = np.array(datos['sesgo'], dtype=np.float32)

    def __str__(self) -> str:
        return f"{self.nombre}(entrada={self.input_size}, salida={self.output_size})"

    def __repr__(self) -> str:
        return self.__str__()


class LucIASystem:
    """Sistema coordinador de la arquitectura neuronal LucIA."""

    def __init__(self, nombre: str = "LucIASystem"):
        self.nombre = nombre
        self.neuronas: Dict[str, LucIANeuronBase] = {}
        self.directivas_recibidas: List[str] = []

    def registrar_neurona(self, neurona: LucIANeuronBase) -> None:
        """Registra una neurona en el mapa del sistema y le asocia la referencia."""
        neurona._system_ref = self
        self.neuronas[neurona.nombre] = neurona
        logger.info(f"Neurona {neurona.nombre} registrada en {self.nombre}")

    def distribuir_directiva_a_todas_las_neuronas(self, directiva: str) -> None:
        """Propaga una directiva global a todas las neuronas registradas."""
        self.directivas_recibidas.append(directiva)
        for nombre, neurona in self.neuronas.items():
            if hasattr(neurona, 'recibir_directiva'):
                neurona.recibir_directiva(directiva)

    def recolectar_todos_los_pesos(self) -> Dict[str, Any]:
        """Recolecta el estado de pesos de todas las neuronas registradas."""
        pesos_totales = {}
        for nombre, neurona in self.neuronas.items():
            pesos_totales[nombre] = neurona.exportar_pesos()
        return pesos_totales

    def cargar_todos_los_pesos(self, pesos_dict: Dict[str, Any]) -> None:
        """Carga pesos a las neuronas registradas."""
        for nombre, datos in pesos_dict.items():
            if nombre in self.neuronas:
                self.neuronas[nombre].cargar_pesos(datos)
