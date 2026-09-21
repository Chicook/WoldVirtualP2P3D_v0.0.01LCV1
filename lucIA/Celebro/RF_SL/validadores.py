"""RF_SL.validadores - Consolidado de las 10 clases utiles de _legado/RF_SL (RFSL2-10).

Reutiliza el codigo legado (identico x9) en un solo fichero: elimina ~90
ficheros duplicados y mantiene la API (nombres de clase + factory).
"""
import numpy as np
import logging
from lucIA.Celebro.RF_SL import NeuronaMemoriaBase

logger = logging.getLogger(__name__)

class _ValidadorBase(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "Validador"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
    def inicializar_pesos(self) -> None:
        self.pesos = np.random.randn(self.input_size, self.output_size).astype(np.float32) * 0.1
        self.sesgo = np.zeros((1, self.output_size), dtype=np.float32)
    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None:
            self.inicializar_pesos()
        salida = np.dot(entrada, self.pesos) + self.sesgo
        self.historial_activaciones.append(salida.copy())
        return salida
    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray):
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'norma': float(np.linalg.norm(grad_pesos))})
        return grad_pesos, grad_sesgo
    def obtener_estadisticas_completas(self):
        stats = self.obtener_estadisticas()
        if self.pesos is not None:
            stats['pesos_mean'] = float(np.mean(self.pesos))
            stats['pesos_std'] = float(np.std(self.pesos))
        return stats
    def verificar_estabilidad(self):
        ok = {'pesos_inicializados': self.pesos is not None}
        if self.pesos is not None:
            ok['pesos_no_explosivos'] = float(np.max(np.abs(self.pesos))) < 10.0
            ok['pesos_no_nan'] = not bool(np.any(np.isnan(self.pesos)))
        return ok
    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size})"

class OutputValidator(_ValidadorBase): pass
class FormatCorrector(_ValidadorBase): pass
class ResponseOptimizer(_ValidadorBase): pass
class QualityController(_ValidadorBase): pass
class NoiseFilter(_ValidadorBase): pass
class OutputNormalizer(_ValidadorBase): pass
class AnomalyDetector(_ValidadorBase): pass
class AdaptiveEnhancer(_ValidadorBase): pass
class PatternLearner(_ValidadorBase): pass
class IntegratedOutputSystem(_ValidadorBase): pass

REGISTRO = {c.__name__: c for c in (OutputValidator, FormatCorrector, ResponseOptimizer, QualityController, NoiseFilter, OutputNormalizer, AnomalyDetector, AdaptiveEnhancer, PatternLearner, IntegratedOutputSystem)}

def crear_validador(nombre: str, input_size: int = 4, output_size: int = 8):
    cls = REGISTRO.get(nombre)
    if cls is None:
        raise KeyError(f"Validador desconocido: {nombre}")
    return cls(input_size=input_size, output_size=output_size, nombre=nombre)
