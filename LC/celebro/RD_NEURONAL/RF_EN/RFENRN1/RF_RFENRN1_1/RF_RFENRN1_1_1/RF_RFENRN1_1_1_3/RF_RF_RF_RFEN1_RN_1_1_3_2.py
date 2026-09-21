try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Módulo dummy para cuando numpy no está disponible
    np = None  # type: ignore


class NeRFMiniNeuron:
    """NeRF mini: MLP simple para color/densidad (preview)."""

    def __init__(self, in_dim: int = 3, hidden: int = 32):
        """Inicializa neurona NeRF mini"""
        if not NUMPY_AVAILABLE:
            # Valores dummy cuando numpy no está disponible
            import random
            random.seed(7)
            self.w1 = [[random.gauss(0, 1 / in_dim) for _ in range(hidden)] for _ in range(in_dim)]
            self.b1 = [random.gauss(0, 0.1) for _ in range(hidden)]
            self.w_sigma = [[random.gauss(0, 1 / hidden)] for _ in range(hidden)]
            self.w_rgb = [[random.gauss(0, 1 / hidden) for _ in range(3)] for _ in range(hidden)]
            self.in_dim = in_dim
            self.hidden = hidden
        else:
            # Versión completa con numpy
            rng = np.random.default_rng(7)  # type: ignore
            self.w1 = (rng.standard_normal((in_dim, hidden)) * (1.0 / in_dim)).astype(np.float32)  # type: ignore
            self.b1 = (rng.standard_normal((hidden,)) * 0.1).astype(np.float32)  # type: ignore
            self.w_sigma = (rng.standard_normal((hidden, 1)) * (1.0 / hidden)).astype(np.float32)  # type: ignore
            self.w_rgb = (rng.standard_normal((hidden, 3)) * (1.0 / hidden)).astype(np.float32)  # type: ignore
            self.in_dim = in_dim
            self.hidden = hidden

    def forward(self, x):
        """Forward pass simplificado"""
        if not NUMPY_AVAILABLE:
            # Versión simplificada sin numpy - solo retorna dummy output
            if isinstance(x, (list, tuple)):
                n = len(x) if isinstance(x[0], (int, float)) else len(x)
            else:
                n = 1
            # Retornar dummy output (N, 4) - [r, g, b, sigma]
            return [[0.5, 0.5, 0.5, 0.5] for _ in range(n)]
        else:
            # Versión completa con numpy
            x = np.atleast_2d(x).astype(np.float32)  # type: ignore
            h = np.maximum(0, x @ self.w1 + self.b1)  # type: ignore
            sigma = np.clip(h @ self.w_sigma, 0, 10)  # type: ignore
            rgb = 1 / (1 + np.exp(-(h @ self.w_rgb)))  # type: ignore
            return np.concatenate([rgb, sigma], axis=1)  # type: ignore  # (N,4)
