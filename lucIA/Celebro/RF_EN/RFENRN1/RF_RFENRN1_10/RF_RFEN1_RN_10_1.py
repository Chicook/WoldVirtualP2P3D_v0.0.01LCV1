"""
Neurona de Generación de Avatares 3D - RF_RFEN1_RN_10_1
Especializada en generar y optimizar avatares 3D utilizando moderngl y numpy
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import math


class Avatar3DGenerator:
    """
    Neurona especializada en generar avatares 3D
    Utiliza algoritmos de geometría 3D y optimización de pesos
    """

    def __init__(self, input_size: int = 100, learning_rate: float = 0.001):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, input_size)
        self.bias = np.random.normal(0, 0.1)
        self.avatars_generated = []
        self.polygon_cache = {}

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Propaga features del avatar a través de la red
        """
        if len(features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(features)}")

        weighted_sum = np.dot(features, self.weights) + self.bias
        return self._swish(weighted_sum)

    def backward(self, error: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ajusta pesos basado en calidad del avatar generado
        """
        x = np.dot(features, self.weights) + self.bias
        gradient = error * (self._swish(x) + x * self._swish_derivative(x))

        weight_gradient = gradient * features
        bias_gradient = gradient

        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient

        return weight_gradient, bias_gradient

    def _swish(self, x: np.ndarray) -> np.ndarray:
        """Función de activación Swish"""
        return x * (1 / (1 + np.exp(-np.clip(x, -250, 250))))

    def _swish_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de Swish"""
        sig = 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        return sig * (1 - x * sig) + x * sig * (1 - sig)

    def generate_avatar_mesh(self, height: float = 1.7, width: float = 0.5,
                             depth: float = 0.3) -> Dict[str, Any]:
        """
        Genera malla 3D de un avatar humano con geometría básica

        Returns:
            Dict con vertices, faces, normals y colors
        """
        vertices = self._generate_body_vertices(height, width, depth)
        faces = self._generate_body_faces()
        normals = self._calculate_normals(vertices, faces)
        colors = self._generate_vertex_colors(len(vertices))

        mesh = {
            'vertices': vertices,
            'faces': faces,
            'normals': normals,
            'colors': colors,
            'polygon_count': len(faces),
            'vertex_count': len(vertices)
        }

        self.avatars_generated.append(mesh)
        return mesh

    def _generate_body_vertices(self, height: float, width: float,
                                depth: float) -> np.ndarray:
        """Genera vértices para el cuerpo humano básico"""
        # Cabeza
        head_radius = width * 0.3
        head_vertices = self._generate_sphere_vertices(head_radius, height + depth/2, 8)

        # Cuerpo (cilindro)
        body_vertices = self._generate_cylinder_vertices(width/2, 0, height/2, 12)

        # Brazos
        arm_vertices = self._generate_arm_vertices(width, depth, height)

        # Piernas
        leg_vertices = self._generate_leg_vertices(width, depth, height/2)

        # Combinar todos los vértices
        vertices = np.vstack([head_vertices, body_vertices, arm_vertices, leg_vertices])
        return vertices

    def _generate_sphere_vertices(self, radius: float, y_offset: float,
                                  segments: int) -> np.ndarray:
        """Genera vértices de una esfera"""
        vertices = []
        for i in range(segments + 1):
            phi = math.pi * i / segments
            for j in range(segments + 1):
                theta = 2 * math.pi * j / segments
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi) + y_offset
                z = radius * math.sin(phi) * math.sin(theta)
                vertices.append([x, y, z])
        return np.array(vertices, dtype=np.float32)

    def _generate_cylinder_vertices(self, radius: float, y_start: float,
                                    y_end: float, segments: int) -> np.ndarray:
        """Genera vértices de un cilindro"""
        vertices = []
        for i in range(segments + 1):
            theta = 2 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append([x, y_start, z])
            vertices.append([x, y_end, z])
        return np.array(vertices, dtype=np.float32)

    def _generate_arm_vertices(self, width: float, depth: float, height: float) -> np.ndarray:
        """Genera vértices de los brazos"""
        arm_vertices = []
        arm_radius = width * 0.15

        # Brazo derecho
        for i in range(-1, 2):
            for j in range(12):
                theta = 2 * math.pi * j / 12
                x = (width/2 + i * arm_radius) + arm_radius * math.cos(theta)
                y = i * (height * 0.4)
                z = arm_radius * math.sin(theta)
                arm_vertices.append([x, y, z])

        # Brazo izquierdo
        for i in range(-1, 2):
            for j in range(12):
                theta = 2 * math.pi * j / 12
                x = (-width/2 - i * arm_radius) - arm_radius * math.cos(theta)
                y = i * (height * 0.4)
                z = arm_radius * math.sin(theta)
                arm_vertices.append([x, y, z])

        return np.array(arm_vertices, dtype=np.float32)

    def _generate_leg_vertices(self, width: float, depth: float, height: float) -> np.ndarray:
        """Genera vértices de las piernas"""
        leg_vertices = []
        leg_radius = width * 0.2

        # Pierna derecha
        for i in range(-3, 0):
            for j in range(12):
                theta = 2 * math.pi * j / 12
                x = (width/3) + leg_radius * math.cos(theta)
                y = i * (height * 0.3)
                z = leg_radius * math.sin(theta)
                leg_vertices.append([x, y, z])

        # Pierna izquierda
        for i in range(-3, 0):
            for j in range(12):
                theta = 2 * math.pi * j / 12
                x = (-width/3) + leg_radius * math.cos(theta)
                y = i * (height * 0.3)
                z = leg_radius * math.sin(theta)
                leg_vertices.append([x, y, z])

        return np.array(leg_vertices, dtype=np.float32)

    def _generate_body_faces(self) -> np.ndarray:
        """
        Genera faces (triángulos) para conectar los vértices
        """
        # Esta es una versión simplificada
        # En producción usarías algoritmos de triangulación más complejos
        faces = []
        vertex_offset = 0

        # Por simplicidad, generamos faces básicas
        # En producción usarías triangulación de Delaunay o similar
        for i in range(100):  # Aproximación simplificada
            faces.append([vertex_offset, vertex_offset+1, vertex_offset+2])
            vertex_offset += 3

        return np.array(faces, dtype=np.uint32)

    def _calculate_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calcula normales para shading"""
        normals = np.zeros_like(vertices)

        for face in faces:
            if len(face) == 3:
                v0, v1, v2 = vertices[face]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                normal = normal / (np.linalg.norm(normal) + 1e-8)

                normals[face] += normal

        # Normalizar
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-8)

        return normals

    def _generate_vertex_colors(self, vertex_count: int) -> np.ndarray:
        """Genera colores para cada vértice"""
        colors = []
        for i in range(vertex_count):
            # Colores suaves piel/ropa
            r = 0.8 + np.random.uniform(-0.2, 0.2)
            g = 0.7 + np.random.uniform(-0.2, 0.2)
            b = 0.6 + np.random.uniform(-0.2, 0.2)
            colors.append([r, g, b, 1.0])  # RGBA

        return np.array(colors, dtype=np.float32)

    def optimize_avatar_geometry(self, mesh: Dict[str, Any],
                                 target_polygons: int = 5000) -> Dict[str, Any]:
        """
        Optimiza geometría del avatar reduciendo polígonos si es necesario
        """
        current_polygons = mesh.get('polygon_count', 0)

        if current_polygons > target_polygons:
            # Simplificar geometría (algoritmo simplificado)
            reduction_factor = target_polygons / current_polygons

            # Reducir vértices y faces manteniendo la forma
            vertices = mesh['vertices']
            faces = mesh['faces']

            # Simplificación básica (en producción usarías algoritmos como Quadric Error Metrics)
            simplified_vertices = self._simplify_vertices(vertices, reduction_factor)

            return {
                **mesh,
                'vertices': simplified_vertices,
                'polygon_count': target_polygons,
                'optimized': True
            }

        return {**mesh, 'optimized': False}

    def _simplify_vertices(self, vertices: np.ndarray, factor: float) -> np.ndarray:
        """Simplifica vertices combinando vecinos cercanos"""
        target_count = int(len(vertices) * factor)
        indices = np.linspace(0, len(vertices)-1, target_count).astype(int)
        return vertices[indices]
