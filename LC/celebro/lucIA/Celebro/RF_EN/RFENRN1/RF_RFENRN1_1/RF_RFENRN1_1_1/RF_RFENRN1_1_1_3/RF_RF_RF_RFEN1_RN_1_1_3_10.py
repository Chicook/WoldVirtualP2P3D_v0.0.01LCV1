import numpy as np


class DiffRenderHookNeuron:
    """Differentiable rendering hook preview: guarda matrices de cámara y luz.
    """

    def __init__(self):
        self.camera_intr = np.array([
            [500.0, 0.0, 128.0],
            [0.0, 500.0, 128.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.light_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def forward(self) -> np.ndarray:
        # Resumen vectorial para preview
        return np.concatenate([
            self.camera_intr.reshape(-1),
            self.camera_pose.reshape(-1),
            self.light_dir.reshape(-1),
        ])
