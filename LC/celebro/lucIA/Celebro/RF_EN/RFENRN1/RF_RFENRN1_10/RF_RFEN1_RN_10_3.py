"""
Neurona de Física 3D - RF_RFEN1_RN_10_3
Especializada en simulación de física para entornos 3D
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import math


class Physics3DNeuron:
    """Neurona especializada en física 3D: gravedad, colisiones, movimiento"""

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.physics_objects = []
        self.gravity = np.array([0, -9.8, 0], dtype=np.float32)

    def forward(self, features: np.ndarray) -> np.ndarray:
        if len(features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(features)}")

        weighted_sum = np.dot(features, self.weights) + self.bias
        return self._relu(weighted_sum)

    def backward(self, error: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, float]:
        x = np.dot(features, self.weights) + self.bias
        gradient = error * self._relu_derivative(x)

        self.weights -= self.learning_rate * gradient * features
        self.bias -= self.learning_rate * gradient

        return gradient * features, gradient

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def update_physics(self, delta_time: float) -> None:
        """Actualiza física de todos los objetos"""
        for obj in self.physics_objects:
            if obj.get('velocity') is not None:
                # Aplicar gravedad
                obj['velocity'] += self.gravity * delta_time

                # Actualizar posición
                obj['position'] += obj['velocity'] * delta_time

                # Reducir velocidad (fricción)
                obj['velocity'] *= 0.98

    def add_rigid_body(self, position: np.ndarray, mass: float = 1.0) -> Dict[str, Any]:
        """Crea cuerpo rígido para simulación"""
        body = {
            'position': np.array(position, dtype=np.float32),
            'velocity': np.zeros(3, dtype=np.float32),
            'mass': mass,
            'collision_shape': 'sphere'
        }
        self.physics_objects.append(body)
        return body

    def check_collision(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> Tuple[bool, np.ndarray]:
        """Verifica colisión entre dos objetos"""
        pos1, pos2 = obj1['position'], obj2['position']
        distance = np.linalg.norm(pos1 - pos2)

        collision_radius = 0.5  # Simplificado
        if distance < collision_radius * 2:
            normal = (pos1 - pos2) / (distance + 1e-8)
            return True, normal

        return False, np.zeros(3)
