"""
RF_SL1_3.py - Random Forest Avanzado
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
            'nombre': self.nombre, 'input_size': self.input_size,
            'output_size': self.output_size, 'pasos': self.pasos,
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


class RandomForestOptimizer(NeuronaMemoriaBase):
    def __init__(self, input_size: int = 4, output_size: int = 8, nombre: str = "RandomForestOptimizer"):
        super().__init__(input_size, output_size, nombre)
        self.inicializar_pesos()
        self._n_trees: int = 100
        self._max_depth: int = 10
        self._min_samples_split: int = 2
        self._subsample_ratio: float = 0.8
        self._n_features: Optional[int] = None
        self._bootstrap: bool = True
        self._trees: List[Any] = []
        self._feature_importances: Optional[np.ndarray] = None
        self._oob_score: float = 0.0
        self._is_fitted: bool = False
        self._convergence_history: List[float] = []
        self._training_time: float = 0.0

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
        return float(np.mean((prediccion - objetivo) ** 2))

    def compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self._trees: return 0.0
        n = len(X)
        y_target = y[:, 0] if y.ndim > 1 else y
        predictions = np.zeros(n)
        counts = np.zeros(n)
        for tree in self._trees:
            mask = np.random.random(n) < 0.632
            if mask.sum() == 0: continue
            pred = tree.predict(X) if hasattr(tree, 'predict') else self.forward(X)
            pred_flat = pred.flatten() if pred.ndim == 1 else pred[:, 0]
            predictions[mask] += pred_flat[mask]
            counts[mask] += 1
        valid = counts > 0
        if valid.sum() == 0: return 0.0
        pred_oob = predictions[valid] / counts[valid]
        return float(1.0 - np.mean(np.abs(pred_oob - y_target[valid])))

    def compute_feature_importances(self) -> np.ndarray:
        if self._feature_importances is not None: return self._feature_importances
        importances = np.zeros(self.input_size)
        for tree in self._trees:
            if hasattr(tree, 'feature_importances_'):
                importances += tree.feature_importances_
        if self._trees: importances /= len(self._trees)
        self._feature_importances = importances
        return importances

    def backward(self, gradiente_salida: np.ndarray, entrada: np.ndarray):
        grad_pesos = np.dot(entrada.T, gradiente_salida)
        grad_sesgo = np.sum(gradiente_salida, axis=0, keepdims=True)
        self.historial_gradientes.append({'pesos': grad_pesos.copy(), 'norma': float(np.linalg.norm(grad_pesos))})
        return grad_pesos, grad_sesgo

    def update_weights(self, grad_pesos: np.ndarray, grad_sesgo: np.ndarray) -> None:
        lr = 0.01
        self.pesos -= lr * grad_pesos
        self.sesgo -= lr * grad_sesgo

    def apply_gradient(self, entrada: np.ndarray, objetivo: np.ndarray) -> float:
        pred = self.forward(entrada)
        grad = pred - objetivo
        gp, gs = self.backward(grad, entrada)
        self.update_weights(gp, gs)
        return self.compute_loss(pred, objetivo)

    def train_step(self, entrada: np.ndarray, objetivo: np.ndarray) -> Dict[str, float]:
        loss_before = self.compute_loss(self.forward(entrada), objetivo)
        loss = self.apply_gradient(entrada, objetivo)
        return {'loss_before': loss_before, 'loss_after': loss, 'improvement': loss_before - loss}

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, verbose: bool = False) -> List[float]:
        t0 = time.perf_counter()
        self._trees.clear()
        n = len(X)
        y_target = y[:, 0] if y.ndim > 1 else y
        losses = []
        for epoch in range(epochs):
            tree = DecisionTree(max_depth=self._max_depth, min_samples_split=self._min_samples_split)
            if self._bootstrap:
                idx = np.random.choice(n, size=n, replace=True)
            else:
                idx = np.arange(n)
            tree.fit(X[idx], y_target[idx])
            self._trees.append(tree)
            pred = self._aggregate_predictions(X)
            loss = self.compute_loss(pred, y_target)
            losses.append(loss)
            self._convergence_history.append(loss)
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"RF Epoch {epoch+1}, loss={loss:.6f}")
        self._training_time = time.perf_counter() - t0
        self._is_fitted = True
        self._oob_score = self.compute_oob_score(X, y)
        self._feature_importances = self.compute_feature_importances()
        return losses

    def _aggregate_predictions(self, X: np.ndarray) -> np.ndarray:
        if not self._trees: return np.zeros(len(X))
        preds = np.zeros(len(X))
        for tree in self._trees:
            pred = tree.predict(X) if hasattr(tree, 'predict') else self.forward(X)
            preds += pred.flatten() if pred.ndim == 1 else pred[:, 0]
        return preds / len(self._trees)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pred = self.predict(X)
        if y.ndim > 1 and pred.ndim > 1:
            pred_t = pred[:, 0]
        elif y.ndim > 1:
            pred_t = pred.flatten()[:len(y)]
        else:
            pred_t = pred
        y_target = y[:, 0] if y.ndim > 1 else y
        mse = self.compute_loss(pred_t, y_target)
        mae = float(np.mean(np.abs(pred_t - y_target)))
        accuracy = float(np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))) if y.ndim > 1 and y.shape[1] > 1 else 0.0
        return {'mse': mse, 'mae': mae, 'accuracy': accuracy, 'n_samples': len(X)}

    def obtener_estadisticas(self) -> Dict[str, Any]:
        stats = super().obtener_estadisticas()
        stats['n_trees'] = self._n_trees
        stats['max_depth'] = self._max_depth
        stats['min_samples_split'] = self._min_samples_split
        stats['bootstrap'] = self._bootstrap
        stats['is_fitted'] = self._is_fitted
        stats['oob_score'] = self._oob_score
        stats['n_trees_built'] = len(self._trees)
        stats['training_time'] = self._training_time
        stats['n_features'] = self._n_features or self.input_size
        if self._feature_importances is not None:
            stats['top_feature'] = int(np.argmax(self._feature_importances))
        if self._convergence_history:
            stats['final_loss'] = float(self._convergence_history[-1])
        return stats

    def verificar_estabilidad(self) -> Dict[str, bool]:
        ok = {'pesos_inicializados': self.pesos is not None}
        if self.pesos is not None:
            ok['pesos_no_explosivos'] = float(np.max(np.abs(self.pesos))) < 10.0
            ok['pesos_no_nan'] = not bool(np.any(np.isnan(self.pesos)))
            ok['pesos_no_inf'] = not bool(np.any(np.isinf(self.pesos)))
            ok['estado_entrenado'] = self._is_fitted
        return ok

    def save(self, filepath: str) -> None:
        state = {
            'pesos': self.pesos, 'sesgo': self.sesgo, 'nombre': self.nombre,
            'input_size': self.input_size, 'output_size': self.output_size,
            'n_trees': self._n_trees, 'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split, 'subsample_ratio': self._subsample_ratio,
            'n_features': self._n_features, 'bootstrap': self._bootstrap,
            'is_fitted': self._is_fitted, 'oob_score': self._oob_score,
            'feature_importances': self._feature_importances,
            'pasos': self.pasos, 'convergence_history': self._convergence_history,
        }
        with open(filepath, 'wb') as f: pickle.dump(state, f)

    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f: state = pickle.load(f)
        self.__dict__.update({k: v for k, v in state.items() if k != 'convergence_history'})
        self._convergence_history = state.get('convergence_history', [])

    def get_config(self) -> Dict[str, Any]:
        return {
            'nombre': self.nombre, 'input_size': self.input_size, 'output_size': self.output_size,
            'n_trees': self._n_trees, 'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split, 'bootstrap': self._bootstrap,
        }

    def summary(self) -> str:
        cfg = self.get_config()
        stats = self.obtener_estadisticas()
        return "\n".join([
            f"=== {cfg['nombre']} ===",
            f"Trees: {cfg['n_trees']}, Depth: {cfg['max_depth']}",
            f"Input: {cfg['input_size']}, Output: {cfg['output_size']}",
            f"Fitted: {stats.get('is_fitted', False)}, OOB: {stats.get('oob_score', 0):.4f}",
            f"Trees built: {stats.get('n_trees_built', 0)}, Params: {self.params_count()}",
        ])

    def __str__(self) -> str:
        return f"{self.nombre}(ent={self.input_size},sal={self.output_size})"


class BootstrapSampler:
    def __init__(self, ratio: float = 0.8, random_state: int = 42):
        self.ratio = ratio
        self.rng = np.random.default_rng(random_state)

    def sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        size = max(1, int(n * self.ratio))
        idx = self.rng.choice(n, size=size, replace=True)
        oob = np.setdiff1d(np.arange(n), idx)
        return X[idx], y[idx], oob

    def sample_indices(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        size = max(1, int(n * self.ratio))
        idx = self.rng.choice(n, size=size, replace=True)
        oob = np.setdiff1d(np.arange(n), idx)
        return idx, oob


class FeatureBagger:
    def __init__(self, n_features: Optional[int] = None, random_state: int = 42):
        self.n_features = n_features
        self.rng = np.random.default_rng(random_state)
        self._selected: Optional[np.ndarray] = None

    def select(self, n_total: int) -> np.ndarray:
        nf = min(self.n_features or n_total, n_total)
        self._selected = self.rng.choice(n_total, size=nf, replace=False)
        return self._selected

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._selected is None: raise RuntimeError("No hay features seleccionados")
        return X[:, self._selected]

    def get_mask(self, n_total: int) -> np.ndarray:
        mask = np.zeros(n_total, dtype=bool)
        if self._selected is not None: mask[self._selected] = True
        return mask


class PredictionVoter:
    def __init__(self):
        self._votes: List[np.ndarray] = []
        self._weights: List[float] = []

    def vote(self, prediction: np.ndarray, weight: float = 1.0) -> None:
        self._votes.append(prediction)
        self._weights.append(weight)

    def majority_vote(self) -> np.ndarray:
        if not self._votes: return np.zeros(1)
        stacked = np.array(self._votes)
        mean_pred = np.mean(stacked, axis=0)
        return mean_pred

    def weighted_vote(self) -> np.ndarray:
        if not self._votes: return np.zeros(1)
        w = np.array(self._weights) / max(1e-8, sum(self._weights))
        return np.average(np.array(self._votes), axis=0, weights=w)

    def reset(self) -> None:
        self._votes.clear()
        self._weights.clear()

    @property
    def n_votes(self) -> int: return len(self._votes)


def create_random_forest_optimizer(input_size: int = 4, output_size: int = 8) -> RandomForestOptimizer:
    return RandomForestOptimizer(input_size=input_size, output_size=output_size)


if __name__ == "__main__":
    logger.info("RF_SL1_3.py cargado exitosamente")
from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_3_DecisionTree import DecisionTree  # CLASSPACK
from LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_3_RandomForestTrainer import RandomForestTrainer  # CLASSPACK
