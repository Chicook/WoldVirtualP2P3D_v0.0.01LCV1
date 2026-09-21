import numpy as np


class DiffusionLiteNeuron:
    """Diffusion-lite preview: UNet muy reducido representado por matrices planas.
    forward devuelve una estimación de ruido eps para un batch sintético.
    """

    def __init__(self, dim: int = 32):
        rng = np.random.default_rng(101)
        self.enc_w = (rng.standard_normal((dim, dim)) * (1.0 / dim)).astype(np.float32)
        self.dec_w = (rng.standard_normal((dim, dim)) * (1.0 / dim)).astype(np.float32)
        self.emb_t = (rng.standard_normal((dim,)) * 0.1).astype(np.float32)

    def forward(self, x: np.ndarray, t: int = 10) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        h = np.tanh(x @ self.enc_w + self.emb_t)
        eps = np.tanh(h @ self.dec_w)
        return eps
