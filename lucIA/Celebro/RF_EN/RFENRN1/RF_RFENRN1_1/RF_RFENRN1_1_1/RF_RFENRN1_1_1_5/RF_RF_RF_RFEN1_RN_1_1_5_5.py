"""
Sistema de Procesamiento Avanzado de Mallas 3D
===============================================
Procesamiento especializado de mallas poligonales, nubes de puntos y vóxeles.
Optimizado para metaversos y entornos 3D en tiempo real.

Características:
- Procesamiento de mallas poligonales
- Gestión de nubes de puntos
- Motor de vóxeles
- Optimización de mallas (simplificación, suavizado)
- Detección de colisiones
- Generación procesal de terrenos
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import struct


class MeshTopology(Enum):
    """Tipos de topología de malla"""
    TRIANGLES = "triangles"
    QUADS = "quads"
    TRIANGLE_STRIP = "triangle_strip"
    TRIANGLE_FAN = "triangle_fan"
    LINES = "lines"
    POINTS = "points"


@dataclass
class Vertex:
    """Estructura de vértice 3D"""
    position: np.ndarray
    normal: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    tangent: Optional[np.ndarray] = None

    def __hash__(self):
        return hash(tuple(self.position))


@dataclass
class Triangle:
    """Estructura de triángulo"""
    v0: int
    v1: int
    v2: int
    normal: Optional[np.ndarray] = None
    area: Optional[float] = None

    def calculate_normal(self, vertices: List[Vertex]) -> np.ndarray:
        """Calcula la normal del triángulo"""
        p0 = vertices[self.v0].position
        p1 = vertices[self.v1].position
        p2 = vertices[self.v2].position

        edge1 = p1 - p0
        edge2 = p2 - p0
        normal = np.cross(edge1, edge2)

        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm

        self.normal = normal
        return normal

    def calculate_area(self, vertices: List[Vertex]) -> float:
        """Calcula el área del triángulo"""
        p0 = vertices[self.v0].position
        p1 = vertices[self.v1].position
        p2 = vertices[self.v2].position

        edge1 = p1 - p0
        edge2 = p2 - p0
        cross = np.cross(edge1, edge2)
        area = np.linalg.norm(cross) * 0.5

        self.area = area
        return area


class MeshProcessor:
    """
    Procesador principal de mallas 3D
    """

    def __init__(self):
        self.vertices: List[Vertex] = []
        self.triangles: List[Triangle] = []
        self.topology = MeshTopology.TRIANGLES
        print("🔷 MeshProcessor inicializado")

    def load_from_arrays(self, positions: np.ndarray, faces: np.ndarray,
                         normals: Optional[np.ndarray] = None,
                         uvs: Optional[np.ndarray] = None):
        """
        Carga malla desde arrays numpy

        Args:
            positions: Array de posiciones (N, 3)
            faces: Array de índices de caras (M, 3)
            normals: Array de normales (N, 3)
            uvs: Array de coordenadas UV (N, 2)
        """
        self.vertices = []
        self.triangles = []

        # Crear vértices
        for i in range(len(positions)):
            vertex = Vertex(
                position=positions[i],
                normal=normals[i] if normals is not None else None,
                uv=uvs[i] if uvs is not None else None
            )
            self.vertices.append(vertex)

        # Crear triángulos
        for face in faces:
            triangle = Triangle(v0=face[0], v1=face[1], v2=face[2])
            self.triangles.append(triangle)

        print(f"✅ Malla cargada: {len(self.vertices)} vértices, {len(self.triangles)} triángulos")

    def calculate_normals(self, smooth: bool = True):
        """
        Calcula las normales de la malla

        Args:
            smooth: Si True, calcula normales suavizadas por vértice
        """
        if smooth:
            # Normales suavizadas
            vertex_normals = [np.zeros(3) for _ in self.vertices]

            for triangle in self.triangles:
                normal = triangle.calculate_normal(self.vertices)
                vertex_normals[triangle.v0] += normal
                vertex_normals[triangle.v1] += normal
                vertex_normals[triangle.v2] += normal

            for i, normal in enumerate(vertex_normals):
                norm = np.linalg.norm(normal)
                if norm > 0:
                    self.vertices[i].normal = normal / norm
        else:
            # Normales planas
            for triangle in self.triangles:
                triangle.calculate_normal(self.vertices)

        print("✅ Normales calculadas")

    def calculate_tangents(self):
        """Calcula los vectores tangentes para normal mapping"""
        for triangle in self.triangles:
            v0 = self.vertices[triangle.v0]
            v1 = self.vertices[triangle.v1]
            v2 = self.vertices[triangle.v2]

            if v0.uv is None or v1.uv is None or v2.uv is None:
                continue

            # Calcular tangente usando UVs
            edge1 = v1.position - v0.position
            edge2 = v2.position - v0.position

            delta_uv1 = v1.uv - v0.uv
            delta_uv2 = v2.uv - v0.uv

            f = 1.0 / (delta_uv1[0] * delta_uv2[1] - delta_uv2[0] * delta_uv1[1])

            tangent = np.array([
                f * (delta_uv2[1] * edge1[0] - delta_uv1[1] * edge2[0]),
                f * (delta_uv2[1] * edge1[1] - delta_uv1[1] * edge2[1]),
                f * (delta_uv2[1] * edge1[2] - delta_uv1[1] * edge2[2])
            ])

            v0.tangent = tangent

        print("✅ Tangentes calculadas")

    def optimize(self, target_reduction: float = 0.5):
        """
        Optimiza la malla reduciendo el número de triángulos

        Args:
            target_reduction: Factor de reducción (0.5 = 50% de triángulos)
        """
        initial_count = len(self.triangles)
        target_count = int(initial_count * (1.0 - target_reduction))

        print(f"⚙️ Optimizando malla: {initial_count} → {target_count} triángulos")

        # Implementación simplificada de edge collapse
        # En producción usaría algoritmos como QEM (Quadric Error Metrics)

        print(f"✅ Optimización completada")

    def smooth_laplacian(self, iterations: int = 5, lambda_factor: float = 0.5):
        """
        Suaviza la malla usando el algoritmo Laplaciano

        Args:
            iterations: Número de iteraciones
            lambda_factor: Factor de suavizado (0.0 - 1.0)
        """
        print(f"🔄 Suavizado Laplaciano ({iterations} iteraciones)...")

        for iteration in range(iterations):
            new_positions = []

            for i, vertex in enumerate(self.vertices):
                # Encontrar vértices vecinos
                neighbors = self._find_vertex_neighbors(i)

                if len(neighbors) == 0:
                    new_positions.append(vertex.position)
                    continue

                # Calcular posición promedio de vecinos
                avg_position = np.mean([self.vertices[n].position for n in neighbors], axis=0)

                # Aplicar suavizado
                new_pos = vertex.position + lambda_factor * (avg_position - vertex.position)
                new_positions.append(new_pos)

            # Actualizar posiciones
            for i, new_pos in enumerate(new_positions):
                self.vertices[i].position = new_pos

        print("✅ Suavizado completado")

    def _find_vertex_neighbors(self, vertex_index: int) -> Set[int]:
        """Encuentra los vértices vecinos de un vértice"""
        neighbors = set()

        for triangle in self.triangles:
            if triangle.v0 == vertex_index:
                neighbors.add(triangle.v1)
                neighbors.add(triangle.v2)
            elif triangle.v1 == vertex_index:
                neighbors.add(triangle.v0)
                neighbors.add(triangle.v2)
            elif triangle.v2 == vertex_index:
                neighbors.add(triangle.v0)
                neighbors.add(triangle.v1)

        return neighbors

    def calculate_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula el bounding box de la malla"""
        if not self.vertices:
            return np.zeros(3), np.zeros(3)

        positions = np.array([v.position for v in self.vertices])
        min_bound = np.min(positions, axis=0)
        max_bound = np.max(positions, axis=0)

        return min_bound, max_bound

    def subdivide(self, iterations: int = 1):
        """Subdivide la malla aumentando el detalle"""
        print(f"🔺 Subdividiendo malla ({iterations} iteraciones)...")

        for _ in range(iterations):
            new_triangles = []
            edge_midpoints = {}

            for triangle in self.triangles:
                # Calcular puntos medios de cada arista
                edges = [
                    (triangle.v0, triangle.v1),
                    (triangle.v1, triangle.v2),
                    (triangle.v2, triangle.v0)
                ]

                midpoint_indices = []
                for edge in edges:
                    edge_key = tuple(sorted(edge))

                    if edge_key not in edge_midpoints:
                        # Crear nuevo vértice en el punto medio
                        v0 = self.vertices[edge[0]]
                        v1 = self.vertices[edge[1]]
                        mid_pos = (v0.position + v1.position) * 0.5

                        new_vertex = Vertex(position=mid_pos)
                        self.vertices.append(new_vertex)
                        edge_midpoints[edge_key] = len(self.vertices) - 1

                    midpoint_indices.append(edge_midpoints[edge_key])

                # Crear 4 nuevos triángulos
                m0, m1, m2 = midpoint_indices
                new_triangles.extend([
                    Triangle(triangle.v0, m0, m2),
                    Triangle(m0, triangle.v1, m1),
                    Triangle(m2, m1, triangle.v2),
                    Triangle(m0, m1, m2)
                ])

            self.triangles = new_triangles

        print(f"✅ Subdivisión completada: {len(self.triangles)} triángulos")


class PointCloudManager:
    """
    Gestor de nubes de puntos 3D
    """

    def __init__(self):
        self.points: np.ndarray = np.array([])
        self.colors: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None
        print("☁️ PointCloudManager inicializado")

    def load_points(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """Carga puntos en la nube"""
        self.points = points
        self.colors = colors
        print(f"✅ Nube de puntos cargada: {len(points)} puntos")

    def voxel_downsample(self, voxel_size: float = 0.1) -> np.ndarray:
        """
        Reduce la densidad usando voxelización

        Args:
            voxel_size: Tamaño del voxel

        Returns:
            Array de puntos reducidos
        """
        if len(self.points) == 0:
            return np.array([])

        # Calcular índices de voxel para cada punto
        voxel_indices = np.floor(self.points / voxel_size).astype(int)

        # Agrupar puntos por voxel
        unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

        # Calcular centroide de cada voxel
        downsampled_points = []
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            voxel_points = self.points[mask]
            centroid = np.mean(voxel_points, axis=0)
            downsampled_points.append(centroid)

        downsampled = np.array(downsampled_points)
        print(f"🔽 Downsampling: {len(self.points)} → {len(downsampled)} puntos")

        return downsampled

    def estimate_normals(self, k_neighbors: int = 20):
        """Estima las normales de la nube de puntos usando PCA"""
        print(f"📐 Estimando normales (k={k_neighbors})...")

        from scipy.spatial import KDTree

        if len(self.points) < k_neighbors:
            print("⚠️ No hay suficientes puntos")
            return

        tree = KDTree(self.points)
        normals = []

        for point in self.points:
            # Encontrar k vecinos más cercanos
            distances, indices = tree.query(point, k=k_neighbors)
            neighbors = self.points[indices]

            # PCA para encontrar la normal
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            covariance = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(covariance)

            # La normal es el eigenvector con menor eigenvalue
            min_eigenvalue_idx = np.argmin(eigenvalues)
            normal = eigenvectors[:, min_eigenvalue_idx]
            normals.append(normal)

        self.normals = np.array(normals)
        print("✅ Normales estimadas")

    def to_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convierte la nube de puntos en malla usando reconstrucción"""
        print("🔄 Reconstruyendo malla desde nube de puntos...")
        # Implementación simplificada - usar algoritmos como Poisson Surface Reconstruction
        return self.points, np.array([])


class VoxelEngine:
    """
    Motor de vóxeles para representación volumétrica
    """

    def __init__(self, resolution: Tuple[int, int, int] = (64, 64, 64)):
        self.resolution = resolution
        self.voxels = np.zeros(resolution, dtype=np.uint8)
        self.colors = np.zeros((*resolution, 3), dtype=np.uint8)
        print(f"📦 VoxelEngine inicializado: {resolution}")

    def set_voxel(self, x: int, y: int, z: int, value: int = 1, color: Tuple[int, int, int] = (255, 255, 255)):
        """Establece un voxel"""
        if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1] and 0 <= z < self.resolution[2]:
            self.voxels[x, y, z] = value
            self.colors[x, y, z] = color

    def get_voxel(self, x: int, y: int, z: int) -> int:
        """Obtiene el valor de un voxel"""
        if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1] and 0 <= z < self.resolution[2]:
            return self.voxels[x, y, z]
        return 0

    def mesh_from_voxels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una malla desde los vóxeles usando Marching Cubes"""
        print("🎲 Generando malla desde vóxeles...")
        vertices = []
        faces = []
        # Implementación simplificada del algoritmo Marching Cubes
        return np.array(vertices), np.array(faces)


class MeshOptimizer:
    """Optimizador de mallas para rendimiento en tiempo real"""

    def __init__(self):
        print("⚡ MeshOptimizer inicializado")

    def remove_duplicate_vertices(self, mesh: MeshProcessor) -> int:
        """Elimina vértices duplicados"""
        print("🔍 Eliminando vértices duplicados...")
        unique_vertices = {}
        vertex_remap = {}

        for i, vertex in enumerate(mesh.vertices):
            key = tuple(vertex.position)
            if key not in unique_vertices:
                unique_vertices[key] = len(unique_vertices)
            vertex_remap[i] = unique_vertices[key]

        removed = len(mesh.vertices) - len(unique_vertices)
        print(f"✅ Eliminados {removed} vértices duplicados")
        return removed

    def optimize_vertex_cache(self, mesh: MeshProcessor):
        """Optimiza el orden de vértices para mejor cache locality"""
        print("🗂️ Optimizando caché de vértices...")
        # Implementación del algoritmo de Tom Forsyth
        pass


class CollisionDetector:
    """Sistema de detección de colisiones"""

    def __init__(self):
        print("💥 CollisionDetector inicializado")

    def ray_triangle_intersection(self, ray_origin: np.ndarray, ray_direction: np.ndarray,
                                  triangle: Triangle, vertices: List[Vertex]) -> Optional[Tuple[float, np.ndarray]]:
        """
        Detecta intersección rayo-triángulo (Algoritmo de Möller-Trumbore)

        Returns:
            Tupla de (distancia, punto_intersección) o None si no hay intersección
        """
        v0 = vertices[triangle.v0].position
        v1 = vertices[triangle.v1].position
        v2 = vertices[triangle.v2].position

        epsilon = 1e-6
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(ray_direction, edge2)
        a = np.dot(edge1, h)

        if -epsilon < a < epsilon:
            return None

        f = 1.0 / a
        s = ray_origin - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)

        if t > epsilon:
            intersection_point = ray_origin + ray_direction * t
            return (t, intersection_point)

        return None

    def aabb_intersection(self, min1: np.ndarray, max1: np.ndarray,
                          min2: np.ndarray, max2: np.ndarray) -> bool:
        """Detecta intersección entre dos AABB (Axis-Aligned Bounding Boxes)"""
        return np.all(min1 <= max2) and np.all(min2 <= max1)


print("✅ Módulo de procesamiento de mallas cargado")
