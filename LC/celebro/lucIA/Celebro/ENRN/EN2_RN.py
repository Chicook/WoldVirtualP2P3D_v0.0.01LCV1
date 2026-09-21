import numpy as np
import math
from typing import Tuple, Dict, Any
from . import NeuronaEntradaBase, LUCIA_CONFIG


class NeuronaEntradaXavier(NeuronaEntradaBase):
    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaEntradaXavier",
                 usar_uniforme: bool = True,
                 factor_escala: float = 1.0):
        super().__init__(input_size, output_size, nombre)
        self.usar_uniforme = usar_uniforme
        self.factor_escala = factor_escala
        self.varianza_esperada = 0.0
        self.varianza_real = 0.0

    def inicializar_pesos(self) -> None:
        fan_avg = (self.input_size + self.output_size) / 2.0
        self.varianza_esperada = 1.0 / fan_avg
        if self.usar_uniforme:
            lim = math.sqrt(6.0 / (self.input_size + self.output_size)) * self.factor_escala
            self.pesos = np.random.uniform(-lim, lim, (self.input_size, self.output_size)).astype(LUCIA_CONFIG['precision'])
        else:
            std = math.sqrt(2.0 / (self.input_size + self.output_size)) * self.factor_escala
            self.pesos = np.random.normal(0, std, (self.input_size, self.output_size)).astype(LUCIA_CONFIG['precision'])
        self.sesgo = np.zeros((1, self.output_size), dtype=LUCIA_CONFIG['precision'])
        self.varianza_real = float(np.var(self.pesos))

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None:
            self.inicializar_pesos()
        salida = np.dot(entrada, self.pesos) + self.sesgo
        self.varianza_real = 0.9 * self.varianza_real + 0.1 * float(np.var(salida)) if self.historial_activaciones else float(np.var(salida))
        self.historial_activaciones.append(salida.copy())
        return salida

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'sesgo': grad_sesgo.copy(),
                                          'norma_pesos': np.linalg.norm(grad_pesos), 'norma_sesgo': np.linalg.norm(grad_sesgo)})
        return grad_pesos, grad_sesgo

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        stats = self.obtener_estadisticas()
        stats.update({'varianza_esperada': self.varianza_esperada, 'varianza_real': self.varianza_real,
                      'ratio_varianza': self.varianza_real / max(self.varianza_esperada, 1e-12)})
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        est = {'pesos_inicializados': self.pesos is not None, 'sesgo_inicializado': self.sesgo is not None}
        if self.pesos is not None:
            pmax = float(np.max(np.abs(self.pesos)))
            pmin = max(float(np.min(np.abs(self.pesos))), 1e-12)
            est['pesos_no_explosivos'] = pmax < 5.0
            est['pesos_no_desaparecen'] = pmin > 1e-6
            est['pesos_balanceados'] = pmax / pmin < 1000.0
            est['varianza_estable'] = 0.5 <= self.varianza_real / max(self.varianza_esperada, 1e-12) <= 2.0
        return est

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},init=Xavier)"
