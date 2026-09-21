"""
Neurona Especializada 3: Generación de Texturas y Materiales
Sistema PBR avanzado compatible con OpenSim/Second Life
WoldVirtual3DlucIA v0.6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class TextureType(Enum):
    """Tipos de texturas soportadas"""
    DIFFUSE = "diffuse"
    NORMAL = "normal"
    SPECULAR = "specular"
    ROUGHNESS = "roughness"
    METALLIC = "metallic"
    EMISSION = "emission"
    AMBIENT_OCCLUSION = "ao"
    DISPLACEMENT = "displacement"


class MaterialPreset(Enum):
    """Presets de materiales comunes"""
    SKIN = "skin"
    FABRIC = "fabric"
    METAL = "metal"
    PLASTIC = "plastic"
    WOOD = "wood"
    STONE = "stone"
    GLASS = "glass"
    LEATHER = "leather"


@dataclass
class TextureMap:
    """Mapa de textura con sus propiedades"""
    texture_type: TextureType
    resolution: Tuple[int, int]
    data: np.ndarray
    file_path: Optional[str] = None
    wrap_mode: str = "repeat"
    filter_mode: str = "linear"


@dataclass
class PBRMaterial:
    """Material con propiedades PBR (Physically Based Rendering)"""
    name: str
    base_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5
    emission: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    emission_strength: float = 0.0
    textures: Dict[TextureType, TextureMap] = field(default_factory=dict)
    uv_scale: Tuple[float, float] = (1.0, 1.0)
    uv_offset: Tuple[float, float] = (0.0, 0.0)


class TextureGenerator:
    """
    Generador procedural de texturas de alta calidad
    Compatible con pipelines modernos y legacy (OpenSim/SL)
    """

    def __init__(self, default_resolution: Tuple[int, int] = (1024, 1024)):
        self.default_resolution = default_resolution
        self.noise_cache = {}
        self.texture_library = {}

    def generate_procedural_texture(
        self,
        texture_type: TextureType,
        resolution: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
        parameters: Optional[Dict] = None
    ) -> TextureMap:
        """
        Genera una textura procedural

        Args:
            texture_type: Tipo de textura a generar
            resolution: Resolución de la textura
            seed: Semilla para generación determinística
            parameters: Parámetros adicionales de generación

        Returns:
            TextureMap generado
        """
        resolution = resolution or self.default_resolution
        parameters = parameters or {}

        if seed is not None:
            np.random.seed(seed)

        if texture_type == TextureType.DIFFUSE:
            data = self._generate_diffuse(resolution, parameters)
        elif texture_type == TextureType.NORMAL:
            data = self._generate_normal(resolution, parameters)
        elif texture_type == TextureType.ROUGHNESS:
            data = self._generate_roughness(resolution, parameters)
        elif texture_type == TextureType.METALLIC:
            data = self._generate_metallic(resolution, parameters)
        else:
            data = np.ones((*resolution, 4), dtype=np.float32)

        return TextureMap(
            texture_type=texture_type,
            resolution=resolution,
            data=data
        )

    def _generate_diffuse(self, resolution: Tuple[int, int], params: Dict) -> np.ndarray:
        """Genera textura difusa con ruido procedural"""
        h, w = resolution

        # Color base
        base_color = params.get("color", (0.8, 0.7, 0.6))

        # Generar ruido Perlin para variación
        noise = self._perlin_noise(h, w, scale=params.get("noise_scale", 8))

        # Crear textura RGB + Alpha
        texture = np.zeros((h, w, 4), dtype=np.float32)

        for i in range(3):
            texture[:, :, i] = base_color[i] * (0.8 + 0.2 * noise)

        texture[:, :, 3] = 1.0  # Alpha

        # Añadir detalles finos
        if params.get("add_details", True):
            detail_noise = self._perlin_noise(h, w, scale=32)
            for i in range(3):
                texture[:, :, i] += detail_noise * 0.05

        return np.clip(texture, 0.0, 1.0)

    def _generate_normal(self, resolution: Tuple[int, int], params: Dict) -> np.ndarray:
        """Genera mapa de normales procedural"""
        h, w = resolution

        # Generar altura base
        strength = params.get("strength", 1.0)
        height_map = self._perlin_noise(h, w, scale=params.get("scale", 16))

        # Calcular normales desde mapa de altura
        normal_map = np.zeros((h, w, 4), dtype=np.float32)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # Derivadas por diferencias finitas
                dx = (height_map[y, x + 1] - height_map[y, x - 1]) * strength
                dy = (height_map[y + 1, x] - height_map[y - 1, x]) * strength

                # Normal = (-dx, -dy, 1) normalizado
                normal = np.array([-dx, -dy, 1.0])
                normal = normal / np.linalg.norm(normal)

                # Mapear de [-1,1] a [0,1] para almacenamiento
                normal_map[y, x, 0] = (normal[0] + 1.0) * 0.5  # R
                normal_map[y, x, 1] = (normal[1] + 1.0) * 0.5  # G
                normal_map[y, x, 2] = (normal[2] + 1.0) * 0.5  # B
                normal_map[y, x, 3] = 1.0  # A

        # Bordes con normal neutra (0.5, 0.5, 1.0)
        normal_map[0, :] = [0.5, 0.5, 1.0, 1.0]
        normal_map[-1, :] = [0.5, 0.5, 1.0, 1.0]
        normal_map[:, 0] = [0.5, 0.5, 1.0, 1.0]
        normal_map[:, -1] = [0.5, 0.5, 1.0, 1.0]

        return normal_map

    def _generate_roughness(self, resolution: Tuple[int, int], params: Dict) -> np.ndarray:
        """Genera mapa de rugosidad"""
        h, w = resolution

        base_roughness = params.get("base_roughness", 0.5)
        variation = params.get("variation", 0.2)

        # Ruido para variación
        noise = self._perlin_noise(h, w, scale=16)

        # Crear mapa de rugosidad (escala de grises + alpha)
        roughness_map = np.zeros((h, w, 4), dtype=np.float32)

        value = base_roughness + (noise - 0.5) * variation
        value = np.clip(value, 0.0, 1.0)

        roughness_map[:, :, 0] = value
        roughness_map[:, :, 1] = value
        roughness_map[:, :, 2] = value
        roughness_map[:, :, 3] = 1.0

        return roughness_map

    def _generate_metallic(self, resolution: Tuple[int, int], params: Dict) -> np.ndarray:
        """Genera mapa metálico"""
        h, w = resolution

        metallic_value = params.get("metallic", 0.0)

        # Mapa uniforme o con variación
        metallic_map = np.ones((h, w, 4), dtype=np.float32)
        metallic_map[:, :, 0:3] = metallic_value

        if params.get("add_variation", False):
            noise = self._perlin_noise(h, w, scale=32)
            metallic_map[:, :, 0:3] *= (0.9 + 0.1 * noise[:, :, np.newaxis])

        return np.clip(metallic_map, 0.0, 1.0)

    def _perlin_noise(self, h: int, w: int, scale: int = 10) -> np.ndarray:
        """
        Genera ruido Perlin 2D simplificado

        Args:
            h, w: Dimensiones de la textura
            scale: Escala del ruido

        Returns:
            Array 2D con valores [0, 1]
        """
        cache_key = f"{h}_{w}_{scale}"

        if cache_key in self.noise_cache:
            return self.noise_cache[cache_key]

        # Implementación simplificada de ruido Perlin
        # En producción usar librería como noise o opensimplex

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(a, b, t):
            return a + t * (b - a)

        # Generar grilla de gradientes
        grid_h = h // scale + 2
        grid_w = w // scale + 2

        gradients = np.random.randn(grid_h, grid_w, 2)
        gradients = gradients / (np.linalg.norm(gradients, axis=2, keepdims=True) + 1e-10)

        # Interpolar ruido
        noise = np.zeros((h, w))

        for y in range(h):
            for x in range(w):
                # Posición en grilla
                gx = x / scale
                gy = y / scale

                # Índices de grilla
                ix = int(gx)
                iy = int(gy)

                # Fracciones
                fx = gx - ix
                fy = gy - iy

                # Gradientes de esquinas
                g00 = gradients[iy, ix]
                g01 = gradients[iy, ix + 1]
                g10 = gradients[iy + 1, ix]
                g11 = gradients[iy + 1, ix + 1]

                # Vectores de distancia
                d00 = np.array([fx, fy])
                d01 = np.array([fx - 1, fy])
                d10 = np.array([fx, fy - 1])
                d11 = np.array([fx - 1, fy - 1])

                # Productos punto
                n00 = np.dot(g00, d00)
                n01 = np.dot(g01, d01)
                n10 = np.dot(g10, d10)
                n11 = np.dot(g11, d11)

                # Interpolación bilineal con fade
                u = fade(fx)
                v = fade(fy)

                n0 = lerp(n00, n01, u)
                n1 = lerp(n10, n11, u)
                noise[y, x] = lerp(n0, n1, v)

        # Normalizar a [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)

        self.noise_cache[cache_key] = noise
        return noise

    def generate_skin_texture(
        self,
        skin_tone: Tuple[float, float, float],
        resolution: Tuple[int, int] = (1024, 1024)
    ) -> Dict[TextureType, TextureMap]:
        """
        Genera conjunto de texturas para piel realista

        Args:
            skin_tone: Color base de la piel (RGB)
            resolution: Resolución de texturas

        Returns:
            Diccionario con todas las texturas necesarias
        """
        textures = {}

        # Difusa con variación de tono
        textures[TextureType.DIFFUSE] = self.generate_procedural_texture(
            TextureType.DIFFUSE,
            resolution,
            parameters={
                "color": skin_tone,
                "noise_scale": 12,
                "add_details": True
            }
        )

        # Normal para detalle de piel
        textures[TextureType.NORMAL] = self.generate_procedural_texture(
            TextureType.NORMAL,
            resolution,
            parameters={
                "strength": 0.3,
                "scale": 24
            }
        )

        # Rugosidad variable (más suave en piel)
        textures[TextureType.ROUGHNESS] = self.generate_procedural_texture(
            TextureType.ROUGHNESS,
            resolution,
            parameters={
                "base_roughness": 0.6,
                "variation": 0.15
            }
        )

        # Sin metallic (piel no es metálica)
        textures[TextureType.METALLIC] = self.generate_procedural_texture(
            TextureType.METALLIC,
            resolution,
            parameters={"metallic": 0.0}
        )

        return textures


class MaterialSystem:
    """Sistema de gestión de materiales PBR"""

    def __init__(self):
        self.texture_generator = TextureGenerator()
        self.materials = {}
        self.material_presets = self._initialize_presets()

    def _initialize_presets(self) -> Dict[MaterialPreset, Dict]:
        """Inicializa presets de materiales"""
        return {
            MaterialPreset.SKIN: {
                "base_color": (0.8, 0.7, 0.6, 1.0),
                "metallic": 0.0,
                "roughness": 0.6,
                "subsurface": 0.3
            },
            MaterialPreset.FABRIC: {
                "base_color": (0.6, 0.6, 0.7, 1.0),
                "metallic": 0.0,
                "roughness": 0.8
            },
            MaterialPreset.METAL: {
                "base_color": (0.9, 0.9, 0.9, 1.0),
                "metallic": 1.0,
                "roughness": 0.2
            },
            MaterialPreset.PLASTIC: {
                "base_color": (0.8, 0.2, 0.2, 1.0),
                "metallic": 0.0,
                "roughness": 0.4
            },
            MaterialPreset.WOOD: {
                "base_color": (0.4, 0.3, 0.2, 1.0),
                "metallic": 0.0,
                "roughness": 0.7
            }
        }

    def create_material(
        self,
        name: str,
        preset: Optional[MaterialPreset] = None,
        custom_properties: Optional[Dict] = None
    ) -> PBRMaterial:
        """
        Crea un material PBR

        Args:
            name: Nombre del material
            preset: Preset a usar como base
            custom_properties: Propiedades personalizadas

        Returns:
            PBRMaterial creado
        """
        if preset:
            properties = self.material_presets[preset].copy()
        else:
            properties = {}

        if custom_properties:
            properties.update(custom_properties)

        material = PBRMaterial(
            name=name,
            base_color=properties.get("base_color", (1.0, 1.0, 1.0, 1.0)),
            metallic=properties.get("metallic", 0.0),
            roughness=properties.get("roughness", 0.5),
            emission=properties.get("emission", (0.0, 0.0, 0.0)),
            emission_strength=properties.get("emission_strength", 0.0)
        )

        self.materials[name] = material
        return material

    def add_texture_to_material(
        self,
        material: PBRMaterial,
        texture_type: TextureType,
        texture_map: TextureMap
    ):
        """Añade una textura a un material"""
        material.textures[texture_type] = texture_map

    def export_material_for_opensim(self, material: PBRMaterial) -> Dict:
        """
        Exporta material en formato compatible con OpenSimulator

        Returns:
            Diccionario con formato OpenSim/Second Life
        """
        return {
            "Name": material.name,
            "DiffuseColor": {
                "R": material.base_color[0],
                "G": material.base_color[1],
                "B": material.base_color[2],
                "A": material.base_color[3]
            },
            "SpecularColor": {
                "R": 1.0 - material.roughness,
                "G": 1.0 - material.roughness,
                "B": 1.0 - material.roughness
            },
            "Shininess": int((1.0 - material.roughness) * 128),
            "EmissionColor": {
                "R": material.emission[0] * material.emission_strength,
                "G": material.emission[1] * material.emission_strength,
                "B": material.emission[2] * material.emission_strength
            },
            "Textures": {
                tex_type.value: tex_map.file_path
                for tex_type, tex_map in material.textures.items()
                if tex_map.file_path
            }
        }


class PBRMaterialBuilder:
    """Constructor avanzado de materiales PBR"""

    def __init__(self):
        self.material_system = MaterialSystem()
        self.texture_generator = TextureGenerator()

    def build_complete_material(
        self,
        name: str,
        material_type: MaterialPreset,
        generate_textures: bool = True,
        texture_resolution: Tuple[int, int] = (1024, 1024)
    ) -> PBRMaterial:
        """
        Construye un material completo con texturas

        Args:
            name: Nombre del material
            material_type: Tipo de material
            generate_textures: Si generar texturas procedurales
            texture_resolution: Resolución de texturas

        Returns:
            Material PBR completo
        """
        material = self.material_system.create_material(name, material_type)

        if generate_textures:
            # Generar texturas según tipo de material
            if material_type == MaterialPreset.SKIN:
                textures = self.texture_generator.generate_skin_texture(
                    material.base_color[:3],
                    texture_resolution
                )
            else:
                textures = self._generate_standard_textures(
                    material,
                    texture_resolution
                )

            # Añadir texturas al material
            for tex_type, tex_map in textures.items():
                self.material_system.add_texture_to_material(
                    material,
                    tex_type,
                    tex_map
                )

        return material

    def _generate_standard_textures(
        self,
        material: PBRMaterial,
        resolution: Tuple[int, int]
    ) -> Dict[TextureType, TextureMap]:
        """Genera texturas estándar para un material"""
        textures = {}

        # Difusa basada en color base
        textures[TextureType.DIFFUSE] = self.texture_generator.generate_procedural_texture(
            TextureType.DIFFUSE,
            resolution,
            parameters={"color": material.base_color[:3]}
        )

        # Normal
        textures[TextureType.NORMAL] = self.texture_generator.generate_procedural_texture(
            TextureType.NORMAL,
            resolution
        )

        # Rugosidad
        textures[TextureType.ROUGHNESS] = self.texture_generator.generate_procedural_texture(
            TextureType.ROUGHNESS,
            resolution,
            parameters={"base_roughness": material.roughness}
        )

        # Metálico
        textures[TextureType.METALLIC] = self.texture_generator.generate_procedural_texture(
            TextureType.METALLIC,
            resolution,
            parameters={"metallic": material.metallic}
        )

        return textures
