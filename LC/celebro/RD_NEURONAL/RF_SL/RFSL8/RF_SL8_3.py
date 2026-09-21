"""
RF_SL8_3.py - Optimización de Respuesta Meta-Learning
========================================
"""

import numpy as np
import logging
from lucIA.Celebro.RD_Neuronal.RF_SL import NeuronaMemoriaBase

logger = logging.getLogger(__name__)



class ResponseOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "ResponseOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()

    def inicializar_pesos(self) -> None:
        std = 0.1
        self.pesos = np.random.randn(self.input_size, self.output_size).astype(np.float32) * std
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
        self.historial_gradientes.append({
            'pesos': grad_pesos.copy(),
            'norma': float(np.linalg.norm(grad_pesos)),
        })
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


def create_response_optimizer(input_size: int = 4, output_size: int = 8) -> ResponseOptimizer:
    return ResponseOptimizer(input_size=input_size, output_size=output_size)


if __name__ == "__main__":
    logger.info("RF_SL8_3.py cargado exitosamente")
