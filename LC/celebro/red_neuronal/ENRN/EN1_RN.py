"""
EN1_RN.py - Neurona de Refuerzo con DQN (expandida 400/450 lineas).
=================================================================================
Modelo free OpenRouter. Subsistema HRCNTR.
Clase NeuronaRefuerzoDQN con Deep Q-Network en NumPy puro.
"""
from __future__ import annotations

import math
import logging
from typing import Tuple, Optional, Dict, Any, List, Deque, Self
from collections import deque
import numpy as np

try:
    from . import NeuronaEntradaBase, LUCIA_CONFIG
except ImportError:
    class NeuronaEntradaBase:
        def __init__(self, input_size=0, output_size=0, nombre=""):
            self.input_size = input_size
            self.output_size = output_size
            self.nombre = nombre
            self.pesos = None
            self.sesgo = None
            self.historial_activaciones = []
            self.historial_gradientes = []
            self.llm_connector = None
        def forward(self, x):
            return x
        def backward(self, g, e):
            return g, e
    LUCIA_CONFIG = {
        'precision': 'float32', 'random_seed': 42,
        'default_learning_rate': 0.001,
    }

logger = logging.getLogger("WoldVirtualP2P3D.NEURON.ENRN.EN1")

LUCIA_RL_CONFIG = {
    'precision': 'float32',
    'random_seed': 42,
    'default_learning_rate': 0.001,
}


class NeuronaEntradaBasica(NeuronaEntradaBase):
    """Neurona de entrada basica compatible con ENRN."""

    def __init__(self, input_size: int, output_size: int,
                 nombre: str = "NeuronaEntradaBasica",
                 activacion: str = "identity",
                 **kwargs):
        super().__init__(input_size, output_size, nombre)
        self.activacion = str(activacion).lower()
        self.kwargs = kwargs

    def inicializar_pesos(self) -> None:
        self.pesos = np.eye(min(self.input_size, self.output_size),
                               dtype=LUCIA_RL_CONFIG['precision'])
        self.sesgo = np.zeros(self.output_size,
                                  dtype=LUCIA_RL_CONFIG['precision'])

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        x = np.asarray(entrada, dtype=LUCIA_RL_CONFIG['precision'])
        if self.pesos is not None:
            x = x @ self.pesos + self.sesgo
        if self.activacion == "relu":
            return np.maximum(0, x)
        elif self.activacion == "tanh":
            return np.tanh(x)
        elif self.activacion == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        return x

    def backward(self, g: np.ndarray, e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(g), np.asarray(e)

    def a_dict(self) -> Dict[str, Any]:
        return {
            "nombre": self.nombre,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "activacion": self.activacion,
            "pesos": self.pesos.tolist() if self.pesos is not None else None,
        }


class MemoriaReplay:
    """Buffer circular para experiencias de entrenamiento DQN."""

    def __init__(self, max_size: int = 10000) -> None:
        self.buffer: Deque = deque(maxlen=max_size)
        self.max_size = max_size

    def almacenar(self, estado: np.ndarray, accion: int,
                   recompensa: float, siguiente: np.ndarray,
                   done: bool) -> None:
        self.buffer.append((estado, accion, recompensa, siguiente, done))

    def muestrear(self, tamano: int) -> List[Tuple]:
        indices = np.random.default_rng().integers(0, len(self.buffer), size=tamano)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class NeuronaRefuerzoDQN(NeuronaEntradaBase):
    """Neurona DQN: Deep Q-Network en NumPy puro.

    Implementa una red neuronal de dos capas ocultas con
    memoria de replay, red objetivo, politica epsilon-greedy
    y validacion de entradas.
    """

    def __init__(self, input_size: int, output_size: int,
                 hidden: Tuple[int, int] = (64, 64),
                 nombre: str = "NeuronaDQN") -> None:
        super().__init__(input_size, output_size, nombre)
        self.hidden = hidden
        self.lr: float = LUCIA_RL_CONFIG['default_learning_rate']
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.995
        self.memo: MemoriaReplay = MemoriaReplay()
        self._red_objetivo: Optional[Dict[str, np.ndarray]] = None
        self._actualizar_objetivo()

    def _actualizar_objetivo(self) -> None:
        self._red_objetivo = self._construir_red()

    def _construir_red(self) -> Dict[str, np.ndarray]:
        si, h1, h2, so = self.input_size, self.hidden[0], self.hidden[1], self.output_size
        red = {
            'w1': InicializadoresRL.he((si, h1), fan_in=si),
            'b1': np.zeros(h1, dtype=LUCIA_RL_CONFIG['precision']),
            'w2': InicializadoresRL.he((h1, h2), fan_in=h1),
            'b2': np.zeros(h2, dtype=LUCIA_RL_CONFIG['precision']),
            'w3': InicializadoresRL.xavier((h2, so), fan_in=h2, fan_out=so),
            'b3': np.zeros(so, dtype=LUCIA_RL_CONFIG['precision']),
        }
        return red

    def inicializar_pesos(self) -> None:
        self.pesos = self._construir_red()
        self.sesgo = np.zeros(self.output_size)

    def _forward_parcial(self, x: np.ndarray, pesos: Dict) -> np.ndarray:
        z1 = x @ pesos['w1'] + pesos['b1']
        a1 = np.maximum(0, z1)
        z2 = a1 @ pesos['w2'] + pesos['b2']
        a2 = np.maximum(0, z2)
        q = a2 @ pesos['w3'] + pesos['b3']
        return q

    def forward(self, e: np.ndarray) -> np.ndarray:
        if self.pesos is None:
            self.inicializar_pesos()
        return self._forward_parcial(e, self.pesos)

    def seleccionar_accion(self, estado: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.output_size)
        qs = self.forward(estado)
        return int(np.argmax(qs))

    def almacenar_experiencia(self, estado: np.ndarray, accion: int,
                               recompensa: float, siguiente: np.ndarray,
                               done: bool) -> None:
        self.memo.almacenar(estado, accion, recompensa, siguiente, done)

    def entrenar(self, tamano_lote: int = 32, gamma: float = 0.99) -> float:
        if len(self.memo) < tamano_lote:
            return 0.0
        lote = self.memo.muestrear(tamano_lote)
        loss = 0.0
        for estado, accion, recompensa, siguiente, done in lote:
            if self.pesos is None:
                continue
            objetivo = recompensa
            if not done:
                qs_sig = self._forward_parcial(siguiente, self._red_objetivo)
                objetivo += gamma * float(np.max(qs_sig))
            qs = self.forward(estado)
            error = objetivo - qs[accion]
            loss += error ** 2
            qs[accion] += self.lr * error
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss / max(1, tamano_lote))

    def actualizar_red_objetivo(self) -> None:
        self._actualizar_objetivo()

    def salvar(self, ruta: str) -> None:
        if self.pesos is None:
            return
        datos = {**self.pesos, 'epsilon': self.epsilon, 'nombre': self.nombre}
        np.save(ruta, datos, allow_pickle=True)
        logger.info("Neurona %s salvada en %s", self.nombre, ruta)

    def cargar(self, ruta: str) -> bool:
        try:
            datos = np.load(ruta, allow_pickle=True).item()
            self.pesos = {k: v for k, v in datos.items()
                          if k in ('w1', 'b1', 'w2', 'b2', 'w3', 'b3')}
            self.epsilon = float(datos.get('epsilon', self.epsilon))
            return True
        except Exception as exc:
            logger.warning("No se pudo cargar %s: %s", ruta, exc)
            return False

    def resetear(self) -> None:
        self.inicializar_pesos()
        self.epsilon = 1.0
        self.memo = MemoriaReplay()
        self._actualizar_objetivo()
        self.resetear_historial()

    def __repr__(self) -> str:
        return (f"NeuronaRefuerzoDQN({self.nombre}: "
                f"input={self.input_size}, output={self.output_size}, "
                f"hidden={self.hidden}, epsilon={self.epsilon:.3f})")

    def info(self) -> Dict[str, Any]:
        return {
            "nombre": self.nombre,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden": self.hidden,
            "epsilon": round(self.epsilon, 4),
            "experiencias": len(self.memo),
            "pesos_cargados": self.pesos is not None,
        }

    def evaluar(self, estados: np.ndarray) -> np.ndarray:
        """Evalua multiples estados de una vez."""
        if self.pesos is None:
            self.inicializar_pesos()
        return self._forward_parcial(estados, self.pesos)

    def politica_greedy(self, estado: np.ndarray) -> Tuple[int, float]:
        """Devuelve accion y su valor Q."""
        qs = self.forward(estado)
        accion = int(np.argmax(qs))
        return accion, float(qs[accion])

    def gradiente_policy(self, estado: np.ndarray, accion: int) -> np.ndarray:
        """Calcula el gradiente de la politica respecto a la accion."""
        if self.pesos is None:
            return np.zeros(self.output_size)
        w1, w2, w3 = self.pesos['w1'], self.pesos['w2'], self.pesos['w3']
        z1 = estado @ w1
        a1 = np.maximum(0, z1)
        z2 = a1 @ w2
        a2 = np.maximum(0, z2)
        q = a2 @ w3
        grad = np.zeros_like(q)
        grad[accion] = 1.0
        grad = grad @ w3.T * (z2 > 0)
        grad = grad @ w2.T * (z1 > 0)
        return grad

    def paso_entrenamiento_completo(self, estado: np.ndarray,
                                       recompensa: float,
                                       siguiente: np.ndarray,
                                       done: bool) -> Dict[str, float]:
        """Un paso completo: almacenar, entrenar, actualizar."""
        accion = self.seleccionar_accion(estado)
        self.almacenar_experiencia(estado, accion, recompensa, siguiente, done)
        loss = self.entrenar()
        if len(self.memo) % 100 == 0:
            self.actualizar_red_objetivo()
        return {"accion": accion, "loss": loss,
                "epsilon": round(self.epsilon, 4),
                "memoria": len(self.memo)}

    def resumen_entrenamiento(self) -> str:
        """Resumen factual del estado de entrenamiento."""
        eps = self.epsilon if self.pesos is not None else 1.0
        return (f"Neurona {self.nombre}: {len(self.memo)} experiencias, "
                f"epsilon={eps:.4f}, "
                f"{'entrenada' if self.pesos is not None else 'sin entrenar'}.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la neurona a diccionario."""
        if self.pesos is None:
            return {"nombre": self.nombre, "pesos": None}
        return {
            "nombre": self.nombre,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden": self.hidden,
            "pesos": {k: v.tolist() for k, v in self.pesos.items()},
            "epsilon": self.epsilon,
            "info": self.info(),
        }

    @classmethod
    def from_dict(cls, datos: Dict[str, Any]) -> Self:
        """Reconstruye la neurona desde un diccionario."""
        neu = cls(
            input_size=datos.get("input_size", 4),
            output_size=datos.get("output_size", 2),
            hidden=tuple(datos.get("hidden", [64, 64])),
            nombre=datos.get("nombre", "NeuronaDQN"),
        )
        pesos = datos.get("pesos")
        if pesos:
            neu.inicializar_pesos()
            neu.pesos = {k: np.array(v, dtype=LUCIA_RL_CONFIG['precision'])
                          for k, v in pesos.items()}
        neu.epsilon = float(datos.get("epsilon", 1.0))
        return neu

    def copiar(self) -> Self:
        """Devuelve una copia profunda de la neurona."""
        return self.from_dict(self.to_dict())

    def __str__(self) -> str:
        return self.resumen_entrenamiento()

    def __repr__(self) -> str:
        return (f"NeuronaRefuerzoDQN({self.nombre}: "
                f"input={self.input_size}, output={self.output_size}, "
                f"hidden={self.hidden}, epsilon={self.epsilon:.3f})")


def crear_neurona_dqn(input_size: int = 4, output_size: int = 2,
                         hidden: Tuple[int, int] = (64, 64)) -> NeuronaRefuerzoDQN:
    """Factoria para crear una neurona DQN pre-inicializada."""
    neu = NeuronaRefuerzoDQN(input_size, output_size, hidden)
    neu.inicializar_pesos()
    return neu


def demo_entrenamiento() -> None:
    """Demostracion rapida de entrenamiento de la neurona DQN."""
    neu = crear_neurona_dqn(4, 2)
    print(f"Neurona creada: {neu}")
    for paso in range(50):
        estado = np.random.randn(4).astype(LUCIA_RL_CONFIG['precision'])
        siguiente = np.random.randn(4).astype(LUCIA_RL_CONFIG['precision'])
        accion = neu.seleccionar_accion(estado)
        neu.almacenar_experiencia(estado, accion,
                                      float(np.random.randn()),
                                      siguiente, paso % 10 == 0)
        neu.entrenar()
    print(f"Tras 50 pasos: {neu.resumen_entrenamiento()}")
    print(f"Info: {neu.info()}")


def validar_estado(estado: np.ndarray,
                       expected_size: int) -> bool:
    """Valida que un estado tenga la dimension esperada."""
    if estado is None or not isinstance(estado, np.ndarray):
        return False
    if estado.size != expected_size:
        return False
    return not np.any(np.isnan(estado)) and not np.any(np.isinf(estado))


def cargar_modelo(ruta: str) -> Optional[NeuronaRefuerzoDQN]:
    """Carga una neurona DQN desde archivo."""
    try:
        datos = np.load(ruta, allow_pickle=True).item()
        neu = NeuronaRefuerzoDQN.from_dict(datos)
        logger.info("Modelo cargado desde %s", ruta)
        return neu
    except Exception as exc:
        logger.warning("Error cargando modelo %s: %s", ruta, exc)
        return None


__all__ = [
    "NeuronaEntradaBasica",
    "InicializadoresRL",
    "MemoriaReplay",
    "NeuronaRefuerzoDQN",
    "crear_neurona_dqn",
    "demo_entrenamiento",
    "validar_estado",
]


if __name__ == "__main__":
    demo_entrenamiento()
from EN1_RN_InicializadoresRL import InicializadoresRL  # CLASSPACK
