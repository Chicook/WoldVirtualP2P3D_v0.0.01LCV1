"""
Neurona de Renderizado de Entornos Virtuales - RF_RFEN1_RN_10_2
Especializada en renderizar escenas 3D y entornos virtuales
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import math


class VirtualEnvironmentRenderer:
    """
    Neurona especializada en renderizar entornos virtuales 3D
    Implementa sistemas de cámara, iluminación y rendering optimizado
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.environments_rendered = []
        self.camera_matrices = {}
        self.light_sources = []

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Propaga features del entorno"""
        if len(features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(features)}")

        weighted_sum = np.dot(features, self.weights) + self.bias
        return self._gelu(weighted_sum)

    def backward(self, error: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Ajusta pesos basado en calidad del renderizado"""
        x = np.dot(features, self.weights) + self.bias
        gradient = error * self._gelu_derivative(x)

        weight_gradient = gradient * features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """Función GELU"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _gelu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de GELU"""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def setup_camera(self, position: np.ndarray, target: np.ndarray,
                     up: np.ndarray, fov: float = 60.0,
                     aspect: float = 16/9, near: float = 0.1, far: float = 1000.0) -> Dict[str, Any]:
        """
        Configura cámara virtual para renderizado

        Returns:
            Dict con matrices de view y projection
        """
        view_matrix = self._look_at(position, target, up)
        projection_matrix = self._perspective(fov, aspect, near, far)

        camera = {
            'position': position,
            'target': target,
            'view_matrix': view_matrix,
            'projection_matrix': projection_matrix,
            'view_projection': np.dot(projection_matrix, view_matrix)
        }

        self.camera_matrices['main'] = camera
        return camera

    def _look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Genera matriz LookAt (View Matrix)"""
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)

        z = eye - target
        z = z / np.linalg.norm(z)

        x = np.cross(up, z)
        x = x / np.linalg.norm(x)

        y = np.cross(z, x)

        matrix = np.eye(4, dtype=np.float32)
        matrix[0, :3] = x
        matrix[1, :3] = y
        matrix[2, :3] = z
        matrix[0, 3] = -np.dot(x, eye)
        matrix[1, 3] = -np.dot(y, eye)
        matrix[2, 3] = -np.dot(z, eye)

        return matrix

    def _perspective(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Genera matriz de proyección perspectiva"""
        f = 1.0 / math.tan(math.radians(fov) / 2.0)

        matrix = np.zeros((4, 4), dtype=np.float32)
        matrix[0, 0] = f / aspect
        matrix[1, 1] = f
        matrix[2, 2] = (far + near) / (near - far)
        matrix[2, 3] = (2 * far * near) / (near - far)
        matrix[3, 2] = -1.0

        return matrix

    def add_light_source(self, position: np.ndarray, color: np.ndarray,
                         intensity: float = 1.0, light_type: str = 'point') -> Dict[str, Any]:
        """
        Añade fuente de luz al entorno

        light_type: 'point', 'directional', 'spot'
        """
        light = {
            'position': np.array(position, dtype=np.float32),
            'color': np.array(color, dtype=np.float32),
            'intensity': intensity,
            'type': light_type,
            'direction': None
        }

        if light_type == 'directional':
            light['direction'] = -position / np.linalg.norm(position)

        self.light_sources.append(light)
        return light

    def calculate_lighting(self, position: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Calcula iluminación en un punto usando Phong shading"""
        ambient = 0.1
        final_color = np.array([ambient, ambient, ambient], dtype=np.float32)

        for light in self.light_sources:
            if light['type'] == 'point':
                light_dir = light['position'] - position
                light_dir = light_dir / np.linalg.norm(light_dir)

                # Diffuse
                ndotl = max(0.0, np.dot(normal, light_dir))
                diffuse = light['intensity'] * ndotl * light['color']

                # Specular (simplificado)
                half = (light_dir + np.array([0, 1, 0])) / 2  # Posición de la cámara
                rdotv = max(0.0, np.dot(normal, half))
                specular = light['intensity'] * (rdotv ** 32) * light['color']

                final_color += diffuse + specular

        return np.clip(final_color, 0, 1)

    def render_scene(self, objects: List[Dict[str, Any]],
                     camera: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Renderiza una escena 3D completa

        Returns:
            Dict con renderizado final
        """
        if camera is None and 'main' in self.camera_matrices:
            camera = self.camera_matrices['main']
        elif camera is None:
            camera = self.setup_camera(
                np.array([0, 5, 10], dtype=np.float32),
                np.array([0, 0, 0], dtype=np.float32),
                np.array([0, 1, 0], dtype=np.float32)
            )

        rendered_objects = []

        for obj in objects:
            vertices = obj.get('vertices', [])
            faces = obj.get('faces', [])

            # Transformar vértices al espacio de pantalla
            transformed_vertices = []
            for vertex in vertices:
                v4 = np.append(vertex, 1.0)
                world_pos = np.dot(camera['view_projection'], v4)
                transformed_vertices.append(world_pos)

            rendered_objects.append({
                'transformed_vertices': transformed_vertices,
                'faces': faces,
                'original_obj': obj
            })

        result = {
            'rendered_objects': rendered_objects,
            'camera': camera,
            'lights': len(self.light_sources),
            'polygon_count': sum(len(obj['faces']) for obj in objects)
        }

        self.environments_rendered.append(result)
        return result

    def apply_fog(self, distance: float, fog_density: float = 0.1) -> float:
        """Aplica efecto de fog a una distancia"""
        fog_factor = 1.0 / (distance * fog_density + 1.0)
        return np.clip(fog_factor, 0, 1)

    def cull_back_faces(self, vertices: np.ndarray, faces: np.ndarray,
                        camera_pos: np.ndarray) -> np.ndarray:
        """Elimina caras que están de espaldas a la cámara (back-face culling)"""
        visible_faces = []

        for face in faces:
            if len(face) >= 3:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

                # Calcular normal de la cara
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal = normal / (np.linalg.norm(normal) + 1e-8)

                # Vector a la cámara
                to_camera = camera_pos - v0

                # Si el ángulo es < 90°, la cara es visible
                if np.dot(normal, to_camera) > 0:
                    visible_faces.append(face)

        return np.array(visible_faces, dtype=np.uint32)

    def optimize_for_target_fps(self, target_fps: float = 60.0) -> Dict[str, Any]:
        """
        Optimiza el renderizado para alcanzar FPS objetivo
        """
        current_polygons = sum(
            env.get('polygon_count', 0)
            for env in self.environments_rendered
        )

        if current_polygons > 10000:
            optimization_suggestions = [
                'Reducir LOD de objetos distantes',
                'Activar back-face culling',
                'Reducir sombras dinámicas',
                'Simplificar shaders complejos',
                'Usar occlusion culling'
            ]
        else:
            optimization_suggestions = ['Rendimiento óptimo']

        return {
            'target_fps': target_fps,
            'estimated_fps': 60.0 if current_polygons < 10000 else 30.0,
            'polygon_count': current_polygons,
            'suggestions': optimization_suggestions
        }
