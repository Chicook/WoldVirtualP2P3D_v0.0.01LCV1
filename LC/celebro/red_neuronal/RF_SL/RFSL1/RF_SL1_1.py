"""
RF_SL1_1.py - Gradient Boosting Avanzado (XGBoost/LightGBM)
Optimizado con clases inline y estructura eficiente.
"""

import numpy as np
import logging
import time
import pickle
from typing import Dict, Any, List, Optional, Tuple

class NeuronaMemoriaBase:
    def __init__(self, input_size: int, output_size: int, nombre: str = "Neurona"):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.nombre = str(nombre)
        self.pesos = None
        self.sesgo = None
        self.historial_activaciones: List[np.ndarray] = []
        self.historial_gradientes: List[Dict[str, np.ndarray]] = []
        self.pasos = 0

    def inicializar_pesos(self) -> None:
        raise NotImplementedError

    def forward(self, e: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = {
            'nombre': self.nombre,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'pasos': self.pasos,
            'historial_activaciones_len': len(self.historial_activaciones),
            'historial_gradientes_len': len(self.historial_gradientes),
        }
        if self.pesos is not None:
            stats['pesos_shape'] = self.pesos.shape
            stats['pesos_size'] = self.pesos.size
        if self.sesgo is not None:
            stats['sesgo_shape'] = self.sesgo.shape
        return stats

    def resetear_historial(self) -> None:
        self.historial_activaciones.clear()
        self.historial_gradientes.clear()

    def info(self) -> str:
        return f"{self.nombre}(in={self.input_size},out={self.output_size},pasos={self.pasos})"

    def params_count(self) -> int:
        total = 0
        if self.pesos is not None: total += self.pesos.size
        if self.sesgo is not None: total += self.sesgo.size
        return total

logger = logging.getLogger(__name__)


class GradientBoostingOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "GradientBoostingOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
        self._learning_rate: float = 0.1
        self._n_estimators: int = 100
        self._max_depth: int = 3
        self._subsample: float = 1.0
        self._loss_type: str = "mse"
        self._convergence_history: List[float] = []
        self._timer_train: float = 0.0

    def inicializar_pesos(self) -> None:
        std = 0.1
        self.pesos = np.random.randn(self.input_size, self.output_size).astype(np.float32) * std
        self.sesgo = np.zeros((1, self.output_size), dtype=np.float32)

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        if self.pesos is None: self.inicializar_pesos()
        salida = np.dot(entrada, self.pesos) + self.sesgo
        self.historial_activaciones.append(salida.copy())
        self.pasos += 1
        return salida

    def predict(self, entrada: np.ndarray) -> np.ndarray:
        return self.forward(entrada)

    def compute_loss(self, prediccion: np.ndarray, objetivo: np.ndarray) -> float:
        if self._loss_type == "mse": return float(np.mean((prediccion - objetivo) ** 2))
        diff = prediccion - objetivo
        return float(np.mean(diff * np.log(1 + np.exp(np.abs(diff))) - objetivo * diff))

    def squared_error_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target) ** 2))

    def log_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
        return float(-np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred)))

    def compute_gradient(self, prediccion: np.ndarray, objetivo: np.ndarray) -> np.ndarray:
        diff = prediccion - objetivo
        if self._loss_type == "mse": return diff
        if self._loss_type == "huber":
            delta = 1.0
            abs_diff = np.abs(diff)
            return np.where(abs_diff <= delta, diff, delta * np.sign(diff))
        return diff

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray):
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'norma': float(np.linalg.norm(grad_pesos))})
        return grad_pesos, grad_sesgo

    def update_weights(self, grad_pesos: np.ndarray, grad_sesgo: np.ndarray) -> None:
        self.pesos -= self._learning_rate * grad_pesos
        self.sesgo -= self._learning_rate * grad_sesgo

    def apply_gradient(self, entrada: np.ndarray, objetivo: np.ndarray) -> float:
        pred = self.forward(entrada)
        grad = self.compute_gradient(pred, objetivo)
        gp, gs = self.backward(grad, entrada)
        self.update_weights(gp, gs)
        return self.compute_loss(pred, objetivo)

    def train_step(self, entrada: np.ndarray, objetivo: np.ndarray) -> Dict[str, float]:
        loss_before = self.compute_loss(self.forward(entrada), objetivo)
        loss = self.apply_gradient(entrada, objetivo)
        return {'loss_before': loss_before, 'loss_after': loss, 'improvement': loss_before - loss}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = False) -> List[float]:
        t0 = time.perf_counter()
        losses = []
        for epoch in range(epochs):
            loss = self.apply_gradient(X, y)
            losses.append(loss)
            self._convergence_history.append(loss)
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, loss={loss:.6f}")
        self._timer_train = time.perf_counter() - t0
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pred = self.predict(X)
        mse = self.squared_error_loss(pred, y)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(mse))
        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'n_samples': len(X)}

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = super().obtener_estadisticas()
        stats['learning_rate'] = self._learning_rate
        stats['n_estimators'] = self._n_estimators
        stats['max_depth'] = self._max_depth
        stats['subsample'] = self._subsample
        stats['loss_type'] = self._loss_type
        stats['convergence_history_len'] = len(self._convergence_history)
        stats['timer_train'] = self._timer_train
        if self._convergence_history:
            stats['final_loss'] = float(self._convergence_history[-1])
            stats['initial_loss'] = float(self._convergence_history[0])
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        ok = {'pesos_inicializados': self.pesos is not None}
        if self.pesos is not None:
            ok['pesos_no_explosivos'] = float(np.max(np.abs(self.pesos))) < 10.0
            ok['pesos_no_nan'] = not bool(np.any(np.isnan(self.pesos)))
            ok['pesos_no_inf'] = not bool(np.any(np.isinf(self.pesos)))
            ok['sesgo_no_nan'] = not bool(np.any(np.isnan(self.sesgo)))
        return ok

    def save(self, filepath: str) -> None:
        state = {
            'pesos': self.pesos, 'sesgo': self.sesgo, 'nombre': self.nombre,
            'input_size': self.input_size, 'output_size': self.output_size,
            'learning_rate': self._learning_rate, 'n_estimators': self._n_estimators,
            'max_depth': self._max_depth, 'subsample': self._subsample,
            'loss_type': self._loss_type, 'pasos': self.pasos,
            'convergence_history': self._convergence_history,
        }
        with open(filepath, 'wb') as f: pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f: state = pickle.load(f)
        self.__dict__.update({k: v for k, v in state.items() if k != 'convergence_history'})
        self._convergence_history = state.get('convergence_history', [])

    def get_config(self) -> Dict[str, Any]:
        return {
            'nombre': self.nombre, 'input_size': self.input_size,
            'output_size': self.output_size, 'learning_rate': self._learning_rate,
            'n_estimators': self._n_estimators, 'max_depth': self._max_depth,
            'subsample': self._subsample, 'loss_type': self._loss_type,
        }

    def summary(self) -> str:
        cfg = self.get_config()
        stats = self.obtener_estadisticas()
        return "\n".join([
            f"=== {cfg['nombre']} ===",
            f"Input: {cfg['input_size']}, Output: {cfg['output_size']}",
            f"LR: {cfg['learning_rate']}, Estimators: {cfg['n_estimators']}, Depth: {cfg['max_depth']}",
            f"Pesos: {stats.get('pesos_shape', 'N/A')}, Params: {self.params_count()}",
            f"Pasos: {self.pasos}, Final loss: {stats.get('final_loss', 'N/A')}",
        ])

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size})"


class Sampler:
    def __init__(self, subsample: float = 1.0, random_state: int = 42):
        self.subsample = subsample
        self.rng = np.random.default_rng(random_state)

    def sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.subsample >= 1.0: return X, y
        n = len(X)
        size = max(1, int(n * self.subsample))
        idx = self.rng.choice(n, size=size, replace=False)
        return X[idx], y[idx]

    def sample_indices(self, n: int) -> np.ndarray:
        if self.subsample >= 1.0: return np.arange(n)
        size = max(1, int(n * self.subsample))
        return self.rng.choice(n, size=size, replace=False)


class FeatureSelector:
    def __init__(self, n_features: Optional[int] = None, random_state: int = 42):
        self.n_features = n_features
        self.rng = np.random.default_rng(random_state)
        self._selected: Optional[np.ndarray] = None

    def select(self, X: np.ndarray, n_features: Optional[int] = None) -> np.ndarray:
        nf = n_features or self.n_features or X.shape[1]
        nf = min(nf, X.shape[1])
        self._selected = self.rng.choice(X.shape[1], size=nf, replace=False)
        return self._selected

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._selected is None: raise RuntimeError("No hay features seleccionados")
        return X[:, self._selected]

    def get_mask(self, n_features: int) -> np.ndarray:
        mask = np.zeros(n_features, dtype=bool)
        if self._selected is not None:
            mask[self._selected] = True
        return mask


def create_gradient_boosting_optimizer(input_size: int = 4, output_size: int = 8) -> GradientBoostingOptimizer:
    return GradientBoostingOptimizer(input_size=input_size, output_size=output_size)


def train_booster(X: np.ndarray, y: np.ndarray, n_estimators: int = 50, learning_rate: float = 0.1,
                  max_depth: int = 3) -> GradientBooster:
    booster = GradientBooster(input_size=X.shape[1], output_size=1, n_estimators=n_estimators,
                              learning_rate=learning_rate, max_depth=max_depth)
    booster.fit(X, y)
    return booster


def evaluate_booster(booster: GradientBooster, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = booster.predict(X)
    return {
        'mse': float(np.mean((pred - y) ** 2)),
        'mae': float(np.mean(np.abs(pred - y))),
        'n_estimators': booster.get_n_estimators(),
        'final_loss': booster._training_history[-1] if booster._training_history else 0.0,
    }


if __name__ == "__main__":
    logger.info("RF_SL1_1.py cargado exitosamente")
from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_1_TreeBuilder import TreeBuilder  # CLASSPACK
from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_1_LossCalculator import LossCalculator  # CLASSPACK
from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_1_PredictionAggregator import PredictionAggregator  # CLASSPACK
from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_1_GradientBooster import GradientBooster  # CLASSPACK
