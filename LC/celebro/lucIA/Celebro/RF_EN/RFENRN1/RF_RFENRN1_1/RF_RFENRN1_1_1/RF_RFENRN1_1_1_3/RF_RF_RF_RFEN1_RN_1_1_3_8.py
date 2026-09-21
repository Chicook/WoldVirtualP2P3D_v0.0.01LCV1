import numpy as np


class WaveFunctionCollapseNeuron:
    """WFC preview: matriz de compatibilidad y frecuencias de tiles.
    """

    def __init__(self, tiles: int = 6):
        rng = np.random.default_rng(111)
        self.compat = (rng.uniform(0, 1, (tiles, tiles)) > 0.3).astype(np.float32)
        self.freq = (rng.uniform(0.1, 1.0, (tiles,))).astype(np.float32)
        self.freq /= self.freq.sum()

    def forward(self, length: int = 8) -> np.ndarray:
        # Secuencia probabilística simple como preview
        idx = np.random.choice(len(self.freq), size=(length,), p=(self.freq / self.freq.sum()))
        return idx.astype(np.float32)
