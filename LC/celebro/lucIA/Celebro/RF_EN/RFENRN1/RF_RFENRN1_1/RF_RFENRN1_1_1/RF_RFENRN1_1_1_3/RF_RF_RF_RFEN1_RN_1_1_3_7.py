import numpy as np


class MarchingCubesNeuron:
    """Marching Cubes preview: guarda un grid SDF y devuelve vértices estimados.
    Para preview, forward devuelve resumen numérico (no malla completa) como vector.
    """

    def __init__(self, grid_size: int = 16):
        rng = np.random.default_rng(89)
        self.grid = (rng.standard_normal((grid_size, grid_size, grid_size)) * 0.1).astype(np.float32)
        # insertar esfera SDF aproximada en el centro
        c = (grid_size - 1) / 2.0
        xs = np.arange(grid_size)
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
        sdf = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2) - (grid_size * 0.25)
        self.grid = self.grid + sdf.astype(np.float32) * 0.05

    def forward(self) -> np.ndarray:
        # En preview devolvemos estadísticas como vector (para reportar activación)
        g = self.grid
        stats = np.array([
            float(np.mean(g)),
            float(np.std(g)),
            float(np.min(g)),
            float(np.max(g)),
        ], dtype=np.float32)
        return stats
