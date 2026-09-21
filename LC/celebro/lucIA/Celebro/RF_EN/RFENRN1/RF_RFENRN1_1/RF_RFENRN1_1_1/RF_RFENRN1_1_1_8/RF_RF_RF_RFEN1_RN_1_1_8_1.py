"""
Neurona Especializada 1: Generación de Avatares 3D
Compatible con OpenSimulator, Second Life y metaversos modernos
WoldVirtual3DlucIA v0.6.0
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid


class AvatarType(Enum):
    """Tipos de avatares soportados"""
    HUMANOID = "humanoid"
    ANTHROPOMORPHIC = "anthropomorphic"
    ROBOT = "robot"
    FANTASY = "fantasy"
    CUSTOM = "custom"


class GenderType(Enum):
    """Tipos de género para avatares"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
    CUSTOM = "custom"


@dataclass
class AvatarMorphParameters:
    """Parámetros de morfología del avatar"""
    height: float = 1.75  # metros
    body_fat: float = 0.5  # 0.0 a 1.0
    muscle_tone: float = 0.5  # 0.0 a 1.0
    head_size: float = 1.0  # escala relativa
    eye_size: float = 1.0
    nose_length: float = 1.0
    mouth_width: float = 1.0
    torso_length: float = 1.0
    leg_length: float = 1.0
    arm_length: float = 1.0
    shoulder_width: float = 1.0
    hip_width: float = 1.0
    custom_morphs: Dict[str, float] = field(default_factory=dict)


@dataclass
class AvatarAppearance:
    """Apariencia visual del avatar"""
    skin_color: Tuple[float, float, float] = (0.8, 0.7, 0.6)
    hair_color: Tuple[float, float, float] = (0.3, 0.2, 0.1)
    eye_color: Tuple[float, float, float] = (0.2, 0.4, 0.6)
    hair_style: str = "default"
    facial_hair: str = "none"
    tattoos: List[str] = field(default_factory=list)
    accessories: List[str] = field(default_factory=list)


class AvatarGenerator3D:
    """
    Generador principal de avatares 3D compatible con OpenSim/Second Life
    Utiliza algoritmos modernos de generación procedural
    """

    def __init__(self, quality: str = "high"):
        self.quality = quality
        self.avatar_cache = {}
        self.base_mesh_resolution = self._get_resolution()

    def _get_resolution(self) -> int:
        """Obtiene la resolución del mesh base según calidad"""
        resolutions = {
            "low": 1000,
            "medium": 5000,
            "high": 15000,
            "ultra": 50000
        }
        return resolutions.get(self.quality, 5000)

    def generate_avatar(
        self,
        avatar_type: AvatarType,
        gender: GenderType,
        morphs: AvatarMorphParameters,
        appearance: AvatarAppearance,
        name: Optional[str] = None
    ) -> Dict:
        """
        Genera un avatar 3D completo con todos sus parámetros

        Returns:
            Dict con mesh, texturas, skeleton y metadatos
        """
        avatar_id = str(uuid.uuid4())
        name = name or f"Avatar_{avatar_id[:8]}"

        # Generar geometría base
        base_mesh = self._generate_base_mesh(avatar_type, gender)

        # Aplicar morfología
        morphed_mesh = self._apply_morphology(base_mesh, morphs)

        # Generar skeleton
        skeleton = self._generate_skeleton(avatar_type, morphs)

        # Generar UVs
        uv_map = self._generate_uv_mapping(morphed_mesh)

        # Generar texturas
        textures = self._generate_textures(appearance, uv_map)

        # Generar datos de peso para skinning
        skin_weights = self._generate_skin_weights(morphed_mesh, skeleton)

        avatar_data = {
            "id": avatar_id,
            "name": name,
            "type": avatar_type.value,
            "gender": gender.value,
            "mesh": morphed_mesh,
            "skeleton": skeleton,
            "uv_map": uv_map,
            "textures": textures,
            "skin_weights": skin_weights,
            "morphology": morphs.__dict__,
            "appearance": appearance.__dict__,
            "metadata": self._generate_metadata(avatar_id, name)
        }

        # Cachear avatar
        self.avatar_cache[avatar_id] = avatar_data

        return avatar_data

    def _generate_base_mesh(self, avatar_type: AvatarType, gender: GenderType) -> np.ndarray:
        """Genera el mesh base según tipo y género"""
        vertices_count = self.base_mesh_resolution

        # Geometría procedural básica (simplificada para ejemplo)
        if avatar_type == AvatarType.HUMANOID:
            vertices = self._create_humanoid_mesh(gender, vertices_count)
        elif avatar_type == AvatarType.ROBOT:
            vertices = self._create_robot_mesh(vertices_count)
        else:
            vertices = self._create_generic_mesh(vertices_count)

        return vertices

    def _create_humanoid_mesh(self, gender: GenderType, vertices_count: int) -> np.ndarray:
        """Crea un mesh humanoide base"""
        # Sistema de generación procedural
        vertices = np.zeros((vertices_count, 3))

        # Cabeza (10% superior)
        head_start = int(vertices_count * 0.0)
        head_end = int(vertices_count * 0.1)
        vertices[head_start:head_end] = self._generate_head_vertices(head_end - head_start, gender)

        # Torso (30%)
        torso_start = head_end
        torso_end = int(vertices_count * 0.4)
        vertices[torso_start:torso_end] = self._generate_torso_vertices(
            torso_end - torso_start, gender
        )

        # Brazos (20%)
        arms_start = torso_end
        arms_end = int(vertices_count * 0.6)
        vertices[arms_start:arms_end] = self._generate_arms_vertices(arms_end - arms_start)

        # Piernas (40%)
        legs_start = arms_end
        vertices[legs_start:] = self._generate_legs_vertices(vertices_count - legs_start, gender)

        return vertices

    def _generate_head_vertices(self, count: int, gender: GenderType) -> np.ndarray:
        """Genera vértices de la cabeza"""
        vertices = np.zeros((count, 3))
        radius = 0.12 if gender == GenderType.MALE else 0.11

        for i in range(count):
            theta = (i / count) * 2 * np.pi
            phi = np.random.uniform(0, np.pi)
            vertices[i] = [
                radius * np.sin(phi) * np.cos(theta),
                1.65 + radius * np.cos(phi),
                radius * np.sin(phi) * np.sin(theta)
            ]

        return vertices

    def _generate_torso_vertices(self, count: int, gender: GenderType) -> np.ndarray:
        """Genera vértices del torso"""
        vertices = np.zeros((count, 3))

        for i in range(count):
            t = i / count
            y = 1.65 - t * 0.75
            width = 0.25 if gender == GenderType.MALE else 0.22
            depth = 0.15

            angle = np.random.uniform(0, 2 * np.pi)
            vertices[i] = [
                width * np.cos(angle) * (1 - t * 0.2),
                y,
                depth * np.sin(angle)
            ]

        return vertices

    def _generate_arms_vertices(self, count: int) -> np.ndarray:
        """Genera vértices de brazos"""
        vertices = np.zeros((count, 3))
        arm_count = count // 2

        # Brazo izquierdo
        for i in range(arm_count):
            t = i / arm_count
            vertices[i] = [
                -0.25 - t * 0.05,
                1.4 - t * 0.6,
                np.random.uniform(-0.05, 0.05)
            ]

        # Brazo derecho
        for i in range(arm_count, count):
            t = (i - arm_count) / arm_count
            vertices[i] = [
                0.25 + t * 0.05,
                1.4 - t * 0.6,
                np.random.uniform(-0.05, 0.05)
            ]

        return vertices

    def _generate_legs_vertices(self, count: int, gender: GenderType) -> np.ndarray:
        """Genera vértices de piernas"""
        vertices = np.zeros((count, 3))
        leg_count = count // 2

        # Pierna izquierda
        for i in range(leg_count):
            t = i / leg_count
            vertices[i] = [
                -0.1,
                0.9 - t * 0.9,
                np.random.uniform(-0.05, 0.05)
            ]

        # Pierna derecha
        for i in range(leg_count, count):
            t = (i - leg_count) / leg_count
            vertices[i] = [
                0.1,
                0.9 - t * 0.9,
                np.random.uniform(-0.05, 0.05)
            ]

        return vertices

    def _create_robot_mesh(self, vertices_count: int) -> np.ndarray:
        """Crea un mesh de robot"""
        return np.random.randn(vertices_count, 3) * 0.3

    def _create_generic_mesh(self, vertices_count: int) -> np.ndarray:
        """Crea un mesh genérico"""
        return np.random.randn(vertices_count, 3) * 0.4

    def _apply_morphology(self, mesh: np.ndarray, morphs: AvatarMorphParameters) -> np.ndarray:
        """Aplica parámetros de morfología al mesh"""
        morphed = mesh.copy()

        # Escalar altura
        morphed[:, 1] *= morphs.height / 1.75

        # Aplicar gordura (expandir en X y Z)
        fat_factor = 1.0 + (morphs.body_fat - 0.5) * 0.4
        morphed[:, 0] *= fat_factor
        morphed[:, 2] *= fat_factor

        # Aplicar tono muscular (definición)
        if morphs.muscle_tone > 0.5:
            noise = np.random.randn(*morphed.shape) * 0.01 * (morphs.muscle_tone - 0.5)
            morphed += noise

        return morphed

    def _generate_skeleton(self, avatar_type: AvatarType, morphs: AvatarMorphParameters) -> Dict:
        """Genera el esqueleto del avatar"""
        return {
            "root": {"position": [0, 0, 0], "rotation": [0, 0, 0]},
            "pelvis": {"position": [0, 0.9, 0], "rotation": [0, 0, 0]},
            "spine": {"position": [0, 1.1, 0], "rotation": [0, 0, 0]},
            "chest": {"position": [0, 1.3, 0], "rotation": [0, 0, 0]},
            "neck": {"position": [0, 1.5, 0], "rotation": [0, 0, 0]},
            "head": {"position": [0, 1.65, 0], "rotation": [0, 0, 0]},
            "shoulder_l": {"position": [-0.2, 1.4, 0], "rotation": [0, 0, 0]},
            "shoulder_r": {"position": [0.2, 1.4, 0], "rotation": [0, 0, 0]},
            "upper_arm_l": {"position": [-0.25, 1.2, 0], "rotation": [0, 0, 0]},
            "upper_arm_r": {"position": [0.25, 1.2, 0], "rotation": [0, 0, 0]},
            "forearm_l": {"position": [-0.25, 0.9, 0], "rotation": [0, 0, 0]},
            "forearm_r": {"position": [0.25, 0.9, 0], "rotation": [0, 0, 0]},
            "hand_l": {"position": [-0.25, 0.7, 0], "rotation": [0, 0, 0]},
            "hand_r": {"position": [0.25, 0.7, 0], "rotation": [0, 0, 0]},
            "thigh_l": {"position": [-0.1, 0.6, 0], "rotation": [0, 0, 0]},
            "thigh_r": {"position": [0.1, 0.6, 0], "rotation": [0, 0, 0]},
            "calf_l": {"position": [-0.1, 0.3, 0], "rotation": [0, 0, 0]},
            "calf_r": {"position": [0.1, 0.3, 0], "rotation": [0, 0, 0]},
            "foot_l": {"position": [-0.1, 0.05, 0.05], "rotation": [0, 0, 0]},
            "foot_r": {"position": [0.1, 0.05, 0.05], "rotation": [0, 0, 0]}
        }

    def _generate_uv_mapping(self, mesh: np.ndarray) -> np.ndarray:
        """Genera mapeo UV para texturas"""
        uv_coords = np.zeros((len(mesh), 2))

        for i, vertex in enumerate(mesh):
            u = (np.arctan2(vertex[2], vertex[0]) / (2 * np.pi)) + 0.5
            v = (vertex[1] + 1.0) / 2.5
            uv_coords[i] = [u, v]

        return uv_coords

    def _generate_textures(self, appearance: AvatarAppearance, uv_map: np.ndarray) -> Dict:
        """Genera texturas del avatar"""
        return {
            "diffuse": self._create_diffuse_texture(appearance),
            "normal": "default_normal.png",
            "specular": "default_specular.png",
            "roughness": 0.7,
            "metallic": 0.0
        }

    def _create_diffuse_texture(self, appearance: AvatarAppearance) -> str:
        """Crea textura difusa"""
        texture_hash = hashlib.md5(
            str(appearance.skin_color).encode()
        ).hexdigest()
        return f"skin_texture_{texture_hash}.png"

    def _generate_skin_weights(self, mesh: np.ndarray, skeleton: Dict) -> np.ndarray:
        """Genera pesos de skinning"""
        weights = np.zeros((len(mesh), len(skeleton)))

        for i, vertex in enumerate(mesh):
            # Asignar pesos basados en proximidad a huesos
            for j, (bone_name, bone_data) in enumerate(skeleton.items()):
                bone_pos = np.array(bone_data["position"])
                distance = np.linalg.norm(vertex - bone_pos)
                weights[i, j] = max(0, 1.0 - distance)

            # Normalizar pesos
            if weights[i].sum() > 0:
                weights[i] /= weights[i].sum()

        return weights

    def _generate_metadata(self, avatar_id: str, name: str) -> Dict:
        """Genera metadatos del avatar"""
        return {
            "id": avatar_id,
            "name": name,
            "version": "0.6.0",
            "format": "OpenSim/SecondLife Compatible",
            "created": "2025-11-05",
            "polygon_count": self.base_mesh_resolution,
            "bone_count": 21,
            "texture_resolution": "1024x1024",
            "compatible_platforms": ["OpenSimulator", "Second Life", "WoldVirtual"]
        }


class AvatarMorphologyEngine:
    """Motor de morfología avanzado para avatares"""

    def __init__(self):
        self.morph_presets = self._load_presets()

    def _load_presets(self) -> Dict:
        """Carga presets de morfología"""
        return {
            "athletic": AvatarMorphParameters(
                height=1.80, body_fat=0.15, muscle_tone=0.85
            ),
            "average": AvatarMorphParameters(
                height=1.75, body_fat=0.50, muscle_tone=0.50
            ),
            "heavy": AvatarMorphParameters(
                height=1.70, body_fat=0.85, muscle_tone=0.30
            )
        }

    def apply_preset(self, preset_name: str) -> AvatarMorphParameters:
        """Aplica un preset de morfología"""
        return self.morph_presets.get(preset_name, self.morph_presets["average"])


class ProceduralAvatarBuilder:
    """Constructor procedural de avatares con IA"""

    def __init__(self):
        self.generator = AvatarGenerator3D(quality="high")
        self.morph_engine = AvatarMorphologyEngine()

    def build_from_description(self, description: str) -> Dict:
        """Construye un avatar desde una descripción textual"""
        # Análisis simple de descripción (en producción usar NLP)
        morphs = AvatarMorphParameters()
        appearance = AvatarAppearance()

        if "alto" in description.lower() or "tall" in description.lower():
            morphs.height = 1.85
        if "fuerte" in description.lower() or "strong" in description.lower():
            morphs.muscle_tone = 0.85

        return self.generator.generate_avatar(
            AvatarType.HUMANOID,
            GenderType.NEUTRAL,
            morphs,
            appearance
        )
