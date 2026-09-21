import numpy as np
import math
from typing import Tuple, Dict, Any
try:
    from . import NeuronaEntradaBase, LUCIA_CONFIG
except ImportError:
    import numpy as np
    _np = np
    from typing import Tuple as _Tuple, Dict as _Dict, Any as _Any

    LUCIA_CONFIG = {
        "precision": "float32",
        "random_seed": 42,
        "default_learning_rate": 0.001,
    }

    class NeuronaEntradaBase:
        """Base autocontenida (fallback sin ciclo de import)."""

        def __init__(self, input_size: int, output_size: int, nombre: str = "NeuronaEntrada"):
            self.input_size = input_size
            self.output_size = output_size
            self.nombre = nombre
            self.pesos = None
            self.sesgo = None
            self.historial_activaciones = []
            self.historial_gradientes = []
            self.llm_connector = None

        def inicializar_pesos(self) -> None:
            raise NotImplementedError

        def forward(self, entrada: _np.ndarray) -> _np.ndarray:
            raise NotImplementedError

        def backward(self, g: _np.ndarray, e: _np.ndarray) -> _Tuple[_np.ndarray, _np.ndarray]:
            raise NotImplementedError

        def obtener_estadisticas(self) -> _Dict[str, _Any]:
            return {"nombre": self.nombre, "input_size": self.input_size,
                    "output_size": self.output_size,
                    "activaciones": len(self.historial_activaciones)}

        def obtener_estadisticas_completas(self) -> _Dict[str, _Any]:
            return self.obtener_estadisticas()

        def verificar_estabilidad(self) -> _Dict[str, bool]:
            return {"pesos_inicializados": self.pesos is not None}

        def __str__(self) -> str:
            return f"{self.nombre}(entrada={self.input_size}, salida={self.output_size})"


class NeuronaEntradaTanh(NeuronaEntradaBase):
    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaEntradaTanh",
                 factor_escala: float = 1.0):
        super().__init__(input_size, output_size, nombre)
        self.factor_escala = factor_escala
        self.varianza_real = 0.0
        self.saturacion = False

    def inicializar_pesos(self) -> None:
        std = math.sqrt(2.0 / (self.input_size + self.output_size)) * self.factor_escala
        lim = math.sqrt(6.0 / (self.input_size + self.output_size)) * self.factor_escala
        self.pesos = np.random.uniform(-lim, lim, (self.input_size, self.output_size)).astype(LUCIA_CONFIG['precision'])
        self.sesgo = np.zeros((1, self.output_size), dtype=LUCIA_CONFIG['precision'])

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None:
            self.inicializar_pesos()
        suma = np.dot(entrada, self.pesos) + self.sesgo
        salida = np.tanh(suma)
        if np.sum(np.abs(salida) > 0.95) > salida.size * 0.1:
            self.saturacion = True
        self.varianza_real = 0.9 * self.varianza_real + 0.1 * float(np.var(salida)) if self.historial_activaciones else float(np.var(salida))
        self.historial_activaciones.append(salida.copy())
        return salida

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.historial_activaciones:
            t = self.historial_activaciones[-1]
            grad_act = 1.0 - t ** 2
            gradiente_salida = gradiente_salida * grad_act
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'sesgo': grad_sesgo.copy(),
                                          'norma_pesos': np.linalg.norm(grad_pesos), 'norma_sesgo': np.linalg.norm(grad_sesgo)})
        return grad_pesos, grad_sesgo

    def obtener_estadisticas_completas(self) -> Dict[str, Any]:
        stats = self.obtener_estadisticas()
        stats.update({'varianza_real': self.varianza_real, 'saturacion': self.saturacion})
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        est = {'pesos_inicializados': self.pesos is not None, 'sesgo_inicializado': self.sesgo is not None}
        if self.pesos is not None:
            pmax = float(np.max(np.abs(self.pesos)))
            pmin = max(float(np.min(np.abs(self.pesos))), 1e-12)
            est['pesos_no_explosivos'] = pmax < 5.0
            est['pesos_no_desaparecen'] = pmin > 1e-6
            est['sin_saturacion'] = not self.saturacion
        return est

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},act=Tanh)"
