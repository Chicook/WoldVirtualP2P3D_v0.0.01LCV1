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


class NeuronaEntradaBatchNorm(NeuronaEntradaBase):
    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaEntradaBatchNorm",
                 momentum: float = 0.9,
                 epsilon: float = 1e-5,
                 factor_escala: float = 1.0,
                 modo_entrenamiento: bool = True):
        super().__init__(input_size, output_size, nombre)
        self.momentum = momentum
        self.epsilon = epsilon
        self.factor_escala = factor_escala
        self.modo_entrenamiento = modo_entrenamiento
        self.gamma: np.ndarray = None
        self.beta: np.ndarray = None
        self.running_mean: np.ndarray = None
        self.running_var: np.ndarray = None
        self.varianza_real = 0.0

    def inicializar_pesos(self) -> None:
        std = math.sqrt(2.0 / self.input_size) * self.factor_escala
        self.pesos = np.random.normal(0, std, (self.input_size, self.output_size)).astype(LUCIA_CONFIG['precision'])
        self.sesgo = np.zeros((1, self.output_size), dtype=LUCIA_CONFIG['precision'])
        self.gamma = np.ones((1, self.output_size), dtype=LUCIA_CONFIG['precision'])
        self.beta = np.zeros((1, self.output_size), dtype=LUCIA_CONFIG['precision'])

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None:
            self.inicializar_pesos()
        salida = np.dot(entrada, self.pesos) + self.sesgo
        if self.modo_entrenamiento:
            media_b = np.mean(salida, axis=0, keepdims=True)
            var_b = np.var(salida, axis=0, keepdims=True)
            if self.running_mean is None:
                self.running_mean, self.running_var = media_b, var_b
            else:
                m = self.momentum
                self.running_mean = m * self.running_mean + (1 - m) * media_b
                self.running_var = m * self.running_var + (1 - m) * var_b
            media, varianza = media_b, var_b
        else:
            media = self.running_mean or 0
            varianza = self.running_var or 1
        salida = self.gamma * (salida - media) / np.sqrt(varianza + self.epsilon) + self.beta
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
        stats.update({'varianza_real': self.varianza_real, 'momentum': self.momentum,
                      'running_mean_norm': float(np.linalg.norm(self.running_mean)) if self.running_mean is not None else 0.0})
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        est = {'pesos_inicializados': self.pesos is not None, 'sesgo_inicializado': self.sesgo is not None}
        if self.pesos is not None:
            pmax = float(np.max(np.abs(self.pesos)))
            pmin = max(float(np.min(np.abs(self.pesos))), 1e-12)
            est['pesos_no_explosivos'] = pmax < 5.0
            est['pesos_no_desaparecen'] = pmin > 1e-6
            est['bn_inicializado'] = self.gamma is not None
        return est

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size},bn_mom={self.momentum})"
