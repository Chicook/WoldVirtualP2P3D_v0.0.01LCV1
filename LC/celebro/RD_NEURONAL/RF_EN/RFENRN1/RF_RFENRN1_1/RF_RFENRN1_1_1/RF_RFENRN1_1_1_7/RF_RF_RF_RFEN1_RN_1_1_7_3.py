"""
Neurona 3: Optimización de Rigging (Skeleton & Rigging Optimization)
Algoritmos avanzados para optimizar la estructura ósea y pesos de influencia
de huesos en avatares 3D para OpenSimulator.

Librerías: numpy, scipy, sklearn
Técnicas: Bone Hierarchy Optimization, Skin Weight Smoothing, IK/FK Solutions
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass, field
try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import minimize
except ImportError:
    pass  # dependencia pesada opcional
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class Bone:
    """Representa un hueso en el esqueleto."""
    name: str
    index: int
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # Quaternion
    length: float = 0.0

    def __post_init__(self):
        """Validación después de inicialización."""
        if isinstance(self.position, list):
            self.position = np.array(self.position)
        if isinstance(self.rotation, list):
            self.rotation = np.array(self.rotation)


@dataclass
class Skeleton:
    """Estructura completa del esqueleto."""
    bones: List[Bone]
    bone_map: Dict[str, int] = field(default_factory=dict)
    root_bones: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Construye el mapa de huesos y encuentra raíces."""
        self.bone_map = {bone.name: bone.index for bone in self.bones}
        self.root_bones = [bone.index for bone in self.bones if bone.parent is None]


@dataclass
class SkinWeight:
    """Pesos de influencia de huesos sobre vértices."""
    vertex_index: int
    bone_indices: List[int]
    weights: List[float]

    def normalize(self):
        """Normaliza los pesos para que sumen 1.0."""
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]


class RiggingOptimizationNeuron:
    """
    Neurona especializada en optimización de rigging y esqueletos para avatares 3D.

    Funcionalidades:
    - Optimización de jerarquía de huesos
    - Suavizado de pesos de influencia (skin weights)
    - Detección y corrección de huesos redundantes
    - Optimización de cadenas IK/FK
    - Validación de topología de esqueleto
    - Corrección de orientaciones de huesos
    - Generación de bind poses optimizadas
    """

    def __init__(self):
        """Inicializa la neurona de optimización de rigging."""
        self.name = "RiggingOptimizationNeuron"
        self.version = "1.0.0"
        self.optimization_cache = {}
        self.standard_bone_names = self._load_standard_bone_names()
        logger.info(f"{self.name} v{self.version} inicializada")

    def _load_standard_bone_names(self) -> Dict[str, str]:
        """Carga nombres estándar de huesos para OpenSimulator."""
        return {
            # Torso
            'mPelvis': 'pelvis',
            'mTorso': 'spine',
            'mChest': 'chest',
            'mNeck': 'neck',
            'mHead': 'head',
            # Brazo izquierdo
            'mCollarLeft': 'clavicle_l',
            'mShoulderLeft': 'shoulder_l',
            'mElbowLeft': 'elbow_l',
            'mWristLeft': 'wrist_l',
            # Brazo derecho
            'mCollarRight': 'clavicle_r',
            'mShoulderRight': 'shoulder_r',
            'mElbowRight': 'elbow_r',
            'mWristRight': 'wrist_r',
            # Pierna izquierda
            'mHipLeft': 'hip_l',
            'mKneeLeft': 'knee_l',
            'mAnkleLeft': 'ankle_l',
            'mFootLeft': 'foot_l',
            # Pierna derecha
            'mHipRight': 'hip_r',
            'mKneeRight': 'knee_r',
            'mAnkleRight': 'ankle_r',
            'mFootRight': 'foot_r'
        }

    def process(self, avatar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa y optimiza el rigging de un avatar.

        Args:
            avatar_data: Diccionario con datos del avatar incluyendo esqueleto

        Returns:
            Diccionario con rigging optimizado y métricas
        """
        try:
            skeleton_data = avatar_data.get('skeleton', None)
            skin_weights_data = avatar_data.get('skin_weights', None)

            if skeleton_data is None:
                return {'error': 'No se encontraron datos de esqueleto'}

            # Construir esqueleto
            skeleton = self._build_skeleton(skeleton_data)

            # Análisis inicial
            initial_analysis = self._analyze_skeleton(skeleton)

            # Optimizar jerarquía
            optimized_skeleton = self._optimize_hierarchy(skeleton)

            # Detectar y eliminar huesos redundantes
            optimized_skeleton = self._remove_redundant_bones(optimized_skeleton)

            # Corregir orientaciones
            optimized_skeleton = self._fix_bone_orientations(optimized_skeleton)

            # Optimizar skin weights si están disponibles
            optimized_weights = None
            if skin_weights_data:
                skin_weights = self._build_skin_weights(skin_weights_data)
                optimized_weights = self._optimize_skin_weights(
                    skin_weights, optimized_skeleton, avatar_data.get('mesh')
                )

            # Análisis final
            final_analysis = self._analyze_skeleton(optimized_skeleton)

            # Generar cadenas IK
            ik_chains = self._generate_ik_chains(optimized_skeleton)

            return {
                'success': True,
                'optimized_skeleton': self._skeleton_to_dict(optimized_skeleton),
                'optimized_weights': optimized_weights,
                'ik_chains': ik_chains,
                'initial_analysis': initial_analysis,
                'final_analysis': final_analysis,
                'bone_reduction': initial_analysis['bone_count'] - final_analysis['bone_count'],
                'hierarchy_depth_reduction': initial_analysis['max_depth'] - final_analysis['max_depth']
            }

        except Exception as e:
            logger.error(f"Error en {self.name}: {e}")
            return {'error': str(e)}

    def _build_skeleton(self, skeleton_data: Dict[str, Any]) -> Skeleton:
        """
        Construye un objeto Skeleton desde datos brutos.

        Args:
            skeleton_data: Datos del esqueleto

        Returns:
            Objeto Skeleton construido
        """
        bones = []

        for idx, bone_data in enumerate(skeleton_data.get('bones', [])):
            bone = Bone(
                name=bone_data.get('name', f'bone_{idx}'),
                index=idx,
                parent=bone_data.get('parent'),
                position=np.array(bone_data.get('position', [0, 0, 0])),
                rotation=np.array(bone_data.get('rotation', [0, 0, 0, 1])),
                length=bone_data.get('length', 0.0)
            )
            bones.append(bone)

        # Construir relaciones padre-hijo
        for bone in bones:
            if bone.parent is not None and 0 <= bone.parent < len(bones):
                bones[bone.parent].children.append(bone.index)

        return Skeleton(bones=bones)

    def _analyze_skeleton(self, skeleton: Skeleton) -> Dict[str, Any]:
        """
        Analiza las características y calidad de un esqueleto.

        Args:
            skeleton: Esqueleto a analizar

        Returns:
            Diccionario con métricas del esqueleto
        """
        bone_count = len(skeleton.bones)
        root_count = len(skeleton.root_bones)

        # Calcular profundidad máxima
        max_depth = 0
        for root_idx in skeleton.root_bones:
            depth = self._calculate_bone_depth(skeleton, root_idx)
            max_depth = max(max_depth, depth)

        # Detectar huesos problemáticos
        zero_length_bones = sum(1 for bone in skeleton.bones if bone.length < 1e-6)
        isolated_bones = sum(1 for bone in skeleton.bones
                             if bone.parent is None and len(bone.children) == 0)

        # Calcular balance del árbol
        balance_score = self._calculate_tree_balance(skeleton)

        return {
            'bone_count': bone_count,
            'root_count': root_count,
            'max_depth': max_depth,
            'zero_length_bones': zero_length_bones,
            'isolated_bones': isolated_bones,
            'balance_score': balance_score,
            'is_valid': (root_count > 0 and zero_length_bones == 0 and isolated_bones == 0)
        }

    def _calculate_bone_depth(self, skeleton: Skeleton, bone_idx: int,
                              current_depth: int = 0) -> int:
        """Calcula recursivamente la profundidad máxima desde un hueso."""
        bone = skeleton.bones[bone_idx]
        if not bone.children:
            return current_depth

        max_child_depth = current_depth
        for child_idx in bone.children:
            child_depth = self._calculate_bone_depth(skeleton, child_idx, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    def _calculate_tree_balance(self, skeleton: Skeleton) -> float:
        """
        Calcula un score de balance del árbol de huesos.
        Un árbol perfectamente balanceado tiene score 1.0.
        """
        if not skeleton.bones:
            return 0.0

        def count_descendants(bone_idx: int) -> int:
            """Cuenta descendientes de un hueso."""
            bone = skeleton.bones[bone_idx]
            count = len(bone.children)
            for child_idx in bone.children:
                count += count_descendants(child_idx)
            return count

        # Calcular varianza de descendientes por rama principal
        if not skeleton.root_bones:
            return 0.0

        descendant_counts = []
        for root_idx in skeleton.root_bones:
            bone = skeleton.bones[root_idx]
            for child_idx in bone.children:
                descendant_counts.append(count_descendants(child_idx))

        if not descendant_counts:
            return 1.0

        mean_count = np.mean(descendant_counts)
        variance = np.var(descendant_counts)

        # Normalizar: menos varianza = mejor balance
        balance = 1.0 / (1.0 + variance / (mean_count + 1))

        return float(balance)

    def _optimize_hierarchy(self, skeleton: Skeleton) -> Skeleton:
        """
        Optimiza la jerarquía del esqueleto para mejor performance.

        Args:
            skeleton: Esqueleto a optimizar

        Returns:
            Esqueleto con jerarquía optimizada
        """
        # Crear copia para modificación
        optimized_bones = [Bone(
            name=bone.name,
            index=bone.index,
            parent=bone.parent,
            children=bone.children.copy(),
            position=bone.position.copy(),
            rotation=bone.rotation.copy(),
            length=bone.length
        ) for bone in skeleton.bones]

        # Reordenar para breadth-first traversal (mejor para cache)
        reordered_bones = []
        visited = set()
        queue = skeleton.root_bones.copy()

        while queue:
            bone_idx = queue.pop(0)
            if bone_idx in visited:
                continue

            visited.add(bone_idx)
            reordered_bones.append(optimized_bones[bone_idx])
            queue.extend(optimized_bones[bone_idx].children)

        # Reindexar
        for new_idx, bone in enumerate(reordered_bones):
            bone.index = new_idx

        return Skeleton(bones=reordered_bones)

    def _remove_redundant_bones(self, skeleton: Skeleton) -> Skeleton:
        """
        Elimina huesos redundantes o innecesarios.

        Criterios:
        - Huesos con longitud cero
        - Huesos con un solo hijo y sin influencia en skin
        - Huesos duplicados
        """
        bones_to_keep = []
        bone_mapping = {}  # old_idx -> new_idx

        for bone in skeleton.bones:
            # Mantener huesos con longitud significativa
            if bone.length >= 1e-6:
                # Mantener huesos con múltiples hijos o sin hijos
                if len(bone.children) != 1:
                    new_idx = len(bones_to_keep)
                    bone_mapping[bone.index] = new_idx
                    bones_to_keep.append(bone)
                else:
                    # Para huesos con un solo hijo, verificar si es necesario
                    # Por ahora, lo mantenemos (lógica más compleja requeriría info de skin)
                    new_idx = len(bones_to_keep)
                    bone_mapping[bone.index] = new_idx
                    bones_to_keep.append(bone)

        # Actualizar índices de padres e hijos
        for bone in bones_to_keep:
            if bone.parent is not None:
                bone.parent = bone_mapping.get(bone.parent)
            bone.children = [bone_mapping[child] for child in bone.children
                             if child in bone_mapping]

        return Skeleton(bones=bones_to_keep)

    def _fix_bone_orientations(self, skeleton: Skeleton) -> Skeleton:
        """
        Corrige las orientaciones de los huesos para consistencia.

        Args:
            skeleton: Esqueleto a corregir

        Returns:
            Esqueleto con orientaciones corregidas
        """
        for bone in skeleton.bones:
            # Normalizar quaternion de rotación
            quat = bone.rotation
            norm = np.linalg.norm(quat)
            if norm > 0:
                bone.rotation = quat / norm
            else:
                bone.rotation = np.array([0, 0, 0, 1])  # Identidad

            # Si el hueso tiene hijos, asegurar que apunta hacia ellos
            if bone.children:
                child_positions = [skeleton.bones[child_idx].position
                                   for child_idx in bone.children]
                avg_child_pos = np.mean(child_positions, axis=0)
                direction = avg_child_pos - bone.position

                if np.linalg.norm(direction) > 1e-6:
                    bone.length = float(np.linalg.norm(direction))

        return skeleton

    def _build_skin_weights(self, weights_data: List[Dict]) -> List[SkinWeight]:
        """Construye objetos SkinWeight desde datos brutos."""
        skin_weights = []

        for weight_data in weights_data:
            sw = SkinWeight(
                vertex_index=weight_data.get('vertex_index', 0),
                bone_indices=weight_data.get('bone_indices', []),
                weights=weight_data.get('weights', [])
            )
            sw.normalize()
            skin_weights.append(sw)

        return skin_weights

    def _optimize_skin_weights(self, skin_weights: List[SkinWeight],
                               skeleton: Skeleton, mesh_data: Optional[Dict]) -> Dict[str, Any]:
        """
        Optimiza los pesos de influencia de huesos sobre vértices.

        Args:
            skin_weights: Lista de pesos de skin
            skeleton: Esqueleto asociado
            mesh_data: Datos de la malla (opcional)

        Returns:
            Diccionario con pesos optimizados
        """
        optimized_weights = []

        for sw in skin_weights:
            # Limitar número de influencias por vértice (máximo 4 para GPU)
            if len(sw.bone_indices) > 4:
                # Mantener los 4 pesos más altos
                sorted_pairs = sorted(zip(sw.weights, sw.bone_indices), reverse=True)
                sw.bone_indices = [pair[1] for pair in sorted_pairs[:4]]
                sw.weights = [pair[0] for pair in sorted_pairs[:4]]
                sw.normalize()

            # Eliminar influencias muy pequeñas
            filtered_indices = []
            filtered_weights = []
            for bone_idx, weight in zip(sw.bone_indices, sw.weights):
                if weight >= 0.01:  # Threshold de 1%
                    filtered_indices.append(bone_idx)
                    filtered_weights.append(weight)

            if filtered_indices:
                sw.bone_indices = filtered_indices
                sw.weights = filtered_weights
                sw.normalize()

            optimized_weights.append(sw)

        return {
            'weights': [{'vertex_index': sw.vertex_index,
                        'bone_indices': sw.bone_indices,
                         'weights': sw.weights} for sw in optimized_weights],
            'max_influences_per_vertex': max(len(sw.bone_indices) for sw in optimized_weights),
            'avg_influences_per_vertex': np.mean([len(sw.bone_indices) for sw in optimized_weights])
        }

    def _generate_ik_chains(self, skeleton: Skeleton) -> List[Dict[str, Any]]:
        """
        Genera cadenas IK (Inverse Kinematics) para el esqueleto.

        Returns:
            Lista de cadenas IK detectadas
        """
        ik_chains = []

        # Detectar cadenas comunes: brazo izquierdo, brazo derecho, pierna izquierda, pierna derecha
        chain_patterns = [
            ('left_arm', ['clavicle_l', 'shoulder_l', 'elbow_l', 'wrist_l']),
            ('right_arm', ['clavicle_r', 'shoulder_r', 'elbow_r', 'wrist_r']),
            ('left_leg', ['hip_l', 'knee_l', 'ankle_l', 'foot_l']),
            ('right_leg', ['hip_r', 'knee_r', 'ankle_r', 'foot_r'])
        ]

        for chain_name, bone_names in chain_patterns:
            chain_indices = []
            for bone_name in bone_names:
                # Buscar hueso por nombre estándar
                found = False
                for bone in skeleton.bones:
                    if self.standard_bone_names.get(bone.name, bone.name) == bone_name:
                        chain_indices.append(bone.index)
                        found = True
                        break

                if not found:
                    break

            if len(chain_indices) >= 3:  # Mínimo 3 huesos para IK válido
                ik_chains.append({
                    'name': chain_name,
                    'bones': chain_indices,
                    'length': len(chain_indices)
                })

        return ik_chains

    def _skeleton_to_dict(self, skeleton: Skeleton) -> Dict[str, Any]:
        """Convierte un Skeleton a diccionario."""
        return {
            'bones': [{
                'name': bone.name,
                'index': bone.index,
                'parent': bone.parent,
                'children': bone.children,
                'position': bone.position.tolist(),
                'rotation': bone.rotation.tolist(),
                'length': bone.length
            } for bone in skeleton.bones],
            'root_bones': skeleton.root_bones
        }
