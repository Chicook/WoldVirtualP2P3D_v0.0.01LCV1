import numpy as np


class SirenSDFNeuron:
    """SIREN + SDF preview: pesos mínimos para reporte.
    pesos_w: (in_dim, hidden)
    pesos_b: (hidden,)
    """

    def __init__(self, in_dim: int = 3, hidden: int = 16, w0: float = 30.0):
        self.in_dim = in_dim
        self.hidden = hidden
        self.w0 = w0
        rng = np.random.default_rng(42)
        self.pesos_w = (rng.standard_normal((in_dim, hidden)) * (1.0 / in_dim)).astype(np.float32)
        self.pesos_b = (rng.standard_normal((hidden,)) * 0.1).astype(np.float32)
        # salida final lineal
        self.pesos_out = (rng.standard_normal((hidden, 1)) * (1.0 / hidden)).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        h = np.sin(self.w0 * (x @ self.pesos_w) + self.pesos_b)
        sdf = h @ self.pesos_out  # (N,1)
        return sdf.squeeze(-1)
