import numpy as np


class OptimizerPackNeuron:
    """Optimizers preview: guarda hiperparámetros estilo Lion/Sophia/Adan.
    """

    def __init__(self):
        self.lion = {
            'lr': np.array(1e-3, dtype=np.float32),
            'beta1': np.array(0.9, dtype=np.float32),
            'beta2': np.array(0.99, dtype=np.float32),
        }
        self.sophia = {
            'lr': np.array(2e-4, dtype=np.float32),
            'rho': np.array(0.03, dtype=np.float32),
        }
        self.adan = {
            'lr': np.array(1e-3, dtype=np.float32),
            'beta1': np.array(0.98, dtype=np.float32),
            'beta2': np.array(0.92, dtype=np.float32),
            'beta3': np.array(0.99, dtype=np.float32),
        }

    def forward(self) -> np.ndarray:
        # concatenar hiperparámetros como vector visualizable
        vals = [
            self.lion['lr'], self.lion['beta1'], self.lion['beta2'],
            self.sophia['lr'], self.sophia['rho'],
            self.adan['lr'], self.adan['beta1'], self.adan['beta2'], self.adan['beta3'],
        ]
        return np.array(vals, dtype=np.float32)
