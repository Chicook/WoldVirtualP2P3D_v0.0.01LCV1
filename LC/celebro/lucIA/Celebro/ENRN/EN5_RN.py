import numpy as np
import math
from typing import Tuple, Dict, Any
from . import NeuronaEntradaBase, LUCIA_CONFIG


class NeuronaEntradaRegularizada(NeuronaEntradaBase):
    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaEntradaRegularizada",
                 lambda_l2: float = 0.01,
                 factor_escala: float = 1.0):
        super().__init__(input_size, output_size, nombre)
        self.lambda_l2 = lambda_l2
        self.factor_escala = factor_escala
        self.varianza_real = 0.0
        self.norma_pesos_historial: list = []

    def inicializar_pesos(self) -> None:
        std = math.sqrt(2.0 / self.input_size) * self.factor_escala
        self.pesos = np.random.normal(0, std, (self.input_size, self.output_size)).astype(LUCIA_CONFIG['precision'])
        self.sesgo = np.zeros((1, self.output_size), dtype=LUCIA_CONFIG['precision'])

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None:
            self.inicializar_pesos()
        salida = np.dot(entrada, self.pesos) + self.sesgo
        self.norma_pesos_historial.append(float(np.linalg.norm(self.pesos)))
        self.varianza_real = 0.9 * self.varianza_real + 0.1 * float(np.var(salida)) if self.historial_activaciones else float(np.var(salida))
        self.historial_activaciones.append(salida.copy())
        return salida

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad_pesos = np.dot(entrada.T, gradiente_salida) + self.lambda_l2 * self.pesos
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'sesgo': grad_sesgo.copy(),
                                          'norma_pesos': np.linalg.norm(grad_pesos), 'norma_sesgo': np.linalg.norm(grad_sesgo)})
        return grad_pesos, grad_sesgo

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        stats = self.obtener_estadisticas()
        stats.update({'lambda_l2': self.lambda_l2, 'varianza_real': self.varianza_real,
                      'norma_pesos_prom': np.mean(self.norma_pesos_historial) if self.norma_pesos_historial else 0.0})
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        est = {'pesos_inicializados': self.pesos is not None, 'sesgo_inicializado': self.sesgo is not None}
        if self.pesos is not None:
            pmax = float(np.max(np.abs(self.pesos)))
            pmin = max(float(np.min(np.abs(self.pesos))), 1e-12)
            est['pesos_no_explosivos'] = pmax < 5.0
            est['pesos_no_desaparecen'] = pmin > 1e-6
            est['regularizacion_activa'] = self.lambda_l2 > 0
        return est

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},l2={self.lambda_l2})"
