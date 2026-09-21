"""
Neurona Especializada 2: Procesamiento de Geometría y Meshes 3D
Optimización de polígonos, LOD y procesamiento avanzado
WoldVirtual3DlucIA v0.6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class MeshData:
    """Estructura de datos de mesh"""
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    uvs: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None


class MeshProcessor:
    """
    Procesador avanzado de geometría 3D
    Compatible con formatos OpenSim/Second Life
    """

    def __init__(self):
        self.processing_cache = {}
        self.optimization_level = "balanced"

    def process_mesh(self, mesh_data: MeshData, optimize: bool = True) -> MeshData:
        """
        Procesa y optimiza un mesh 3D

        Args:
            mesh_data: Datos del mesh a procesar
            optimize: Si debe optimizar la geometría

        Returns:
            MeshData procesado
        """
        # Calcular normales si no existen
        if mesh_data.normals is None:
            mesh_data.normals = self.calculate_normals(
                mesh_data.vertices,
                mesh_data.faces
            )

        # Optimizar si es necesario
        if optimize:
            mesh_data = self.optimize_mesh(mesh_data)

        # Validar integridad
        self.validate_mesh(mesh_data)

        return mesh_data

    def calculate_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Calcula normales por vértice

        Args:
            vertices: Array de vértices [N, 3]
            faces: Array de caras [M, 3]

        Returns:
            Normales calculadas [N, 3]
        """
        normals = np.zeros_like(vertices)

        for face in faces:
            # Obtener vértices de la cara
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # Calcular vectores de aristas
            edge1 = v1 - v0
            edge2 = v2 - v0

            # Producto cruz para normal de cara
            face_normal = np.cross(edge1, edge2)
            face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-10)

            # Acumular en normales de vértices
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal

        # Normalizar normales de vértices
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-10)

        return normals

    def optimize_mesh(self, mesh_data: MeshData) -> MeshData:
        """Optimiza la geometría del mesh"""
        # Eliminar vértices duplicados
        mesh_data = self.remove_duplicate_vertices(mesh_data)

        # Eliminar caras degeneradas
        mesh_data = self.remove_degenerate_faces(mesh_data)

        # Optimizar orden de vértices para caché
        mesh_data = self.optimize_vertex_cache(mesh_data)

        return mesh_data

    def remove_duplicate_vertices(self, mesh_data: MeshData, tolerance: float = 1e-6) -> MeshData:
        """Elimina vértices duplicados"""
        vertices = mesh_data.vertices
        faces = mesh_data.faces

        # Encontrar vértices únicos
        unique_vertices = []
        vertex_map = {}
        new_indices = []

        for i, vertex in enumerate(vertices):
            # Crear hash del vértice
            vertex_hash = tuple(np.round(vertex / tolerance).astype(int))

            if vertex_hash not in vertex_map:
                vertex_map[vertex_hash] = len(unique_vertices)
                unique_vertices.append(vertex)

            new_indices.append(vertex_map[vertex_hash])

        # Remapear caras
        new_faces = np.array([[new_indices[f[0]], new_indices[f[1]], new_indices[f[2]]]
                             for f in faces])

        return MeshData(
            vertices=np.array(unique_vertices),
            faces=new_faces,
            normals=mesh_data.normals,
            uvs=mesh_data.uvs,
            colors=mesh_data.colors
        )

    def remove_degenerate_faces(self, mesh_data: MeshData) -> MeshData:
        """Elimina caras degeneradas (área cero)"""
        vertices = mesh_data.vertices
        faces = mesh_data.faces

        valid_faces = []

        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # Calcular área de la cara
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = np.linalg.norm(cross) / 2.0

            # Solo mantener caras con área significativa
            if area > 1e-6:
                valid_faces.append(face)

        mesh_data.faces = np.array(valid_faces)
        return mesh_data

    def optimize_vertex_cache(self, mesh_data: MeshData) -> MeshData:
        """
        Optimiza el orden de vértices para mejor uso de caché GPU
        Implementa algoritmo tipo Forsyth
        """
        # Simplificación del algoritmo para mantener <300 líneas
        # En producción usar implementación completa de Tom Forsyth

        faces = mesh_data.faces
        vertex_count = len(mesh_data.vertices)

        # Calcular valencia de vértices (cuántas caras lo usan)
        vertex_valence = np.zeros(vertex_count, dtype=int)
        for face in faces:
            for vertex_idx in face:
                vertex_valence[vertex_idx] += 1

        # Reordenar caras basándose en valencia
        face_scores = np.array([
            vertex_valence[face[0]] + vertex_valence[face[1]] + vertex_valence[face[2]]
            for face in faces
        ])

        sorted_indices = np.argsort(-face_scores)
        mesh_data.faces = faces[sorted_indices]

        return mesh_data

    def validate_mesh(self, mesh_data: MeshData) -> bool:
        """Valida la integridad del mesh"""
        if len(mesh_data.vertices) == 0:
            raise ValueError("Mesh no tiene vértices")

        if len(mesh_data.faces) == 0:
            raise ValueError("Mesh no tiene caras")

        # Verificar que los índices de caras son válidos
        max_index = np.max(mesh_data.faces)
        if max_index >= len(mesh_data.vertices):
            raise ValueError(f"Índice de cara inválido: {max_index}")

        return True

    def smooth_mesh(self, mesh_data: MeshData, iterations: int = 1, factor: float = 0.5) -> MeshData:
        """
        Suaviza el mesh usando algoritmo Laplacian

        Args:
            mesh_data: Mesh a suavizar
            iterations: Número de iteraciones
            factor: Factor de suavizado (0-1)
        """
        vertices = mesh_data.vertices.copy()
        faces = mesh_data.faces

        # Construir grafo de adyacencia
        adjacency = [set() for _ in range(len(vertices))]
        for face in faces:
            adjacency[face[0]].update([face[1], face[2]])
            adjacency[face[1]].update([face[0], face[2]])
            adjacency[face[2]].update([face[0], face[1]])

        # Aplicar suavizado Laplaciano
        for _ in range(iterations):
            new_vertices = vertices.copy()

            for i in range(len(vertices)):
                if len(adjacency[i]) > 0:
                    # Calcular centroide de vecinos
                    neighbors = list(adjacency[i])
                    centroid = np.mean(vertices[neighbors], axis=0)

                    # Mover vértice hacia centroide
                    new_vertices[i] = vertices[i] + factor * (centroid - vertices[i])

            vertices = new_vertices

        mesh_data.vertices = vertices
        return mesh_data

    def subdivide_mesh(self, mesh_data: MeshData, method: str = "simple") -> MeshData:
        """
        Subdivide el mesh para mayor densidad

        Args:
            mesh_data: Mesh a subdividir
            method: Método de subdivisión ('simple', 'loop', 'catmull-clark')
        """
        if method == "simple":
            return self._simple_subdivision(mesh_data)
        elif method == "loop":
            return self._loop_subdivision(mesh_data)
        else:
            return mesh_data

    def _simple_subdivision(self, mesh_data: MeshData) -> MeshData:
        """Subdivisión simple por punto medio"""
        vertices = list(mesh_data.vertices)
        faces = mesh_data.faces
        new_faces = []

        edge_midpoints = {}

        def get_midpoint(v1, v2):
            """Obtiene o crea punto medio de arista"""
            edge = tuple(sorted([v1, v2]))
            if edge not in edge_midpoints:
                midpoint = (vertices[v1] + vertices[v2]) / 2.0
                edge_midpoints[edge] = len(vertices)
                vertices.append(midpoint)
            return edge_midpoints[edge]

        # Subdividir cada cara en 4
        for face in faces:
            v0, v1, v2 = face

            # Obtener puntos medios
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)

            # Crear 4 nuevas caras
            new_faces.extend([
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20]
            ])

        return MeshData(
            vertices=np.array(vertices),
            faces=np.array(new_faces),
            normals=None,
            uvs=mesh_data.uvs,
            colors=mesh_data.colors
        )

    def _loop_subdivision(self, mesh_data: MeshData) -> MeshData:
        """Subdivisión Loop (más suave)"""
        # Implementación simplificada
        # En producción usar implementación completa de Charles Loop
        return self._simple_subdivision(mesh_data)


class GeometryOptimizer:
    """Optimizador de geometría para rendimiento en tiempo real"""

    def __init__(self):
        self.target_poly_count = 50000

    def optimize_for_platform(
        self,
        mesh_data: MeshData,
        platform: str = "desktop"
    ) -> MeshData:
        """
        Optimiza geometría según plataforma objetivo

        Args:
            mesh_data: Mesh a optimizar
            platform: 'desktop', 'mobile', 'vr'
        """
        target_polys = {
            "mobile": 10000,
            "desktop": 50000,
            "vr": 30000
        }

        self.target_poly_count = target_polys.get(platform, 50000)

        current_poly_count = len(mesh_data.faces)

        if current_poly_count > self.target_poly_count:
            mesh_data = self.decimate_mesh(mesh_data, self.target_poly_count)

        return mesh_data

    def decimate_mesh(self, mesh_data: MeshData, target_count: int) -> MeshData:
        """
        Reduce el número de polígonos del mesh
        Implementa simplificación por colapso de aristas
        """
        vertices = mesh_data.vertices
        faces = list(mesh_data.faces)

        current_count = len(faces)
        reduction_ratio = target_count / current_count

        # Simplificación básica: eliminar caras basándose en importancia
        face_importance = self._calculate_face_importance(vertices, faces)

        # Ordenar por importancia y mantener las más importantes
        sorted_indices = np.argsort(-face_importance)
        keep_count = int(len(faces) * reduction_ratio)
        kept_faces = [faces[i] for i in sorted_indices[:keep_count]]

        mesh_data.faces = np.array(kept_faces)

        # Limpiar vértices no usados
        mesh_data = self._remove_unused_vertices(mesh_data)

        return mesh_data

    def _calculate_face_importance(self, vertices: np.ndarray, faces: List) -> np.ndarray:
        """Calcula importancia de cada cara para decimación"""
        importance = np.zeros(len(faces))

        for i, face in enumerate(faces):
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # Calcular área como medida de importancia
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = np.linalg.norm(np.cross(edge1, edge2)) / 2.0

            # Calcular curvatura aproximada
            centroid = (v0 + v1 + v2) / 3.0
            distance_from_origin = np.linalg.norm(centroid)

            # Importancia = área * factor de curvatura
            importance[i] = area * (1.0 + distance_from_origin)

        return importance

    def _remove_unused_vertices(self, mesh_data: MeshData) -> MeshData:
        """Elimina vértices que no son referenciados por ninguna cara"""
        used_vertices = set()
        for face in mesh_data.faces:
            used_vertices.update(face)

        used_vertices = sorted(used_vertices)

        # Crear mapeo de índices antiguos a nuevos
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}

        # Nuevos vértices
        new_vertices = mesh_data.vertices[used_vertices]

        # Remapear caras
        new_faces = np.array([[vertex_map[f[0]], vertex_map[f[1]], vertex_map[f[2]]]
                             for f in mesh_data.faces])

        return MeshData(
            vertices=new_vertices,
            faces=new_faces,
            normals=mesh_data.normals,
            uvs=mesh_data.uvs,
            colors=mesh_data.colors
        )


class LODGenerator:
    """Generador de niveles de detalle (Level of Detail)"""

    def __init__(self):
        self.lod_levels = 4
        self.optimizer = GeometryOptimizer()

    def generate_lod_chain(self, mesh_data: MeshData) -> List[MeshData]:
        """
        Genera cadena de LODs para un mesh

        Returns:
            Lista de meshes con diferentes niveles de detalle
        """
        lod_chain = [mesh_data]  # LOD 0 (máximo detalle)

        base_poly_count = len(mesh_data.faces)

        for level in range(1, self.lod_levels):
            reduction_factor = 0.5 ** level
            target_polys = int(base_poly_count * reduction_factor)

            lod_mesh = self.optimizer.decimate_mesh(
                MeshData(
                    vertices=mesh_data.vertices.copy(),
                    faces=mesh_data.faces.copy(),
                    normals=mesh_data.normals,
                    uvs=mesh_data.uvs,
                    colors=mesh_data.colors
                ),
                target_polys
            )

            lod_chain.append(lod_mesh)

        return lod_chain

    def calculate_lod_distances(self, object_size: float) -> List[float]:
        """
        Calcula distancias de cambio de LOD

        Args:
            object_size: Tamaño aproximado del objeto en metros

        Returns:
            Lista de distancias para cada nivel LOD
        """
        base_distance = object_size * 10

        distances = []
        for level in range(self.lod_levels):
            distances.append(base_distance * (2 ** level))

        return distances
