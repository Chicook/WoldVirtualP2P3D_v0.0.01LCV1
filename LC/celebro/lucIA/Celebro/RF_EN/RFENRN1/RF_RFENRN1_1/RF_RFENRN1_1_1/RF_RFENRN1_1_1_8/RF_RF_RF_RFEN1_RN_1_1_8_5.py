"""
Neurona Especializada 5: Generación de Terreno e Islas Virtuales
Sistemas procedurales para mundos 3D tipo OpenSim/Second Life
WoldVirtual3DlucIA v0.6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class BiomeType(Enum):
    """Tipos de biomas"""
    OCEAN = "ocean"
    BEACH = "beach"
    PLAINS = "plains"
    FOREST = "forest"
    MOUNTAINS = "mountains"
    DESERT = "desert"
    SNOW = "snow"
    JUNGLE = "jungle"


class TerrainQuality(Enum):
    """Calidad del terreno"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class TerrainConfig:
    """Configuración de generación de terreno"""
    size_x: int = 256
    size_y: int = 256
    height_scale: float = 100.0
    sea_level: float = 0.3
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    seed: Optional[int] = None


@dataclass
class IslandConfig:
    """Configuración de isla"""
    radius: float = 128.0
    center: Tuple[float, float] = (128.0, 128.0)
    falloff_strength: float = 3.0
    beach_width: float = 10.0


class TerrainGenerator:
    """
    Generador procedural de terreno
    Compatible con formato de terreno OpenSimulator
    """

    def __init__(self, config: Optional[TerrainConfig] = None):
        self.config = config or TerrainConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        self.heightmap = None
        self.biome_map = None
        self.water_map = None

    def generate_terrain(self) -> np.ndarray:
        """
        Genera un terreno completo

        Returns:
            Heightmap 2D con alturas normalizadas [0,1]
        """
        # Generar ruido base multi-octava
        base_noise = self._generate_multi_octave_noise(
            self.config.size_x,
            self.config.size_y,
            self.config.octaves,
            self.config.persistence,
            self.config.lacunarity
        )

        # Normalizar a [0, 1]
        base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())

        # Aplicar curva de distribución de altura
        base_noise = self._apply_height_curve(base_noise)

        self.heightmap = base_noise
        return self.heightmap

    def generate_island(self, island_config: Optional[IslandConfig] = None) -> np.ndarray:
        """
        Genera una isla con máscara radial

        Args:
            island_config: Configuración de la isla

        Returns:
            Heightmap de isla
        """
        island_config = island_config or IslandConfig()

        # Generar terreno base
        terrain = self.generate_terrain()

        # Crear máscara radial para isla
        island_mask = self._create_island_mask(
            self.config.size_x,
            self.config.size_y,
            island_config
        )

        # Aplicar máscara
        island_terrain = terrain * island_mask

        # Asegurar que los bordes estén bajo el nivel del mar
        island_terrain = self._ensure_water_edges(island_terrain, island_config)

        self.heightmap = island_terrain
        return island_terrain

    def _generate_multi_octave_noise(
        self,
        width: int,
        height: int,
        octaves: int,
        persistence: float,
        lacunarity: float
    ) -> np.ndarray:
        """Genera ruido Perlin multi-octava"""
        noise_map = np.zeros((height, width))

        max_amplitude = 0.0
        amplitude = 1.0
        frequency = 1.0

        for octave in range(octaves):
            octave_noise = self._generate_perlin_noise(
                width,
                height,
                int(4 * frequency)
            )

            noise_map += octave_noise * amplitude

            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        # Normalizar por amplitud máxima
        noise_map /= max_amplitude

        return noise_map

    def _generate_perlin_noise(self, width: int, height: int, scale: int) -> np.ndarray:
        """Genera ruido Perlin 2D"""
        if scale <= 0:
            scale = 1

        # Grilla de gradientes
        grid_width = width // scale + 2
        grid_height = height // scale + 2

        gradients = np.random.randn(grid_height, grid_width, 2)
        gradients /= (np.linalg.norm(gradients, axis=2, keepdims=True) + 1e-10)

        noise = np.zeros((height, width))

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(a, b, t):
            return a + t * (b - a)

        for y in range(height):
            for x in range(width):
                # Posición en grilla
                gx = x / scale
                gy = y / scale

                ix = int(gx)
                iy = int(gy)

                fx = gx - ix
                fy = gy - iy

                # Gradientes de esquinas
                g00 = gradients[iy, ix]
                g01 = gradients[iy, ix + 1]
                g10 = gradients[iy + 1, ix]
                g11 = gradients[iy + 1, ix + 1]

                # Vectores de offset
                d00 = np.array([fx, fy])
                d01 = np.array([fx - 1, fy])
                d10 = np.array([fx, fy - 1])
                d11 = np.array([fx - 1, fy - 1])

                # Productos punto
                n00 = np.dot(g00, d00)
                n01 = np.dot(g01, d01)
                n10 = np.dot(g10, d10)
                n11 = np.dot(g11, d11)

                # Interpolación
                u = fade(fx)
                v = fade(fy)

                n0 = lerp(n00, n01, u)
                n1 = lerp(n10, n11, u)
                noise[y, x] = lerp(n0, n1, v)

        return noise

    def _apply_height_curve(self, heightmap: np.ndarray) -> np.ndarray:
        """Aplica curva de redistribución de alturas"""
        # Curva exponencial para más planicies y montañas pronunciadas
        return np.power(heightmap, 1.5)

    def _create_island_mask(
        self,
        width: int,
        height: int,
        island_config: IslandConfig
    ) -> np.ndarray:
        """Crea máscara radial para isla"""
        mask = np.zeros((height, width))

        cx, cy = island_config.center
        radius = island_config.radius
        falloff = island_config.falloff_strength

        for y in range(height):
            for x in range(width):
                # Distancia al centro
                dx = x - cx
                dy = y - cy
                distance = np.sqrt(dx * dx + dy * dy)

                # Aplicar falloff suave
                if distance < radius:
                    # Función de falloff suave
                    t = distance / radius
                    mask[y, x] = 1.0 - np.power(t, falloff)
                else:
                    mask[y, x] = 0.0

        return mask

    def _ensure_water_edges(
        self,
        heightmap: np.ndarray,
        island_config: IslandConfig
    ) -> np.ndarray:
        """Asegura que los bordes estén bajo el agua"""
        result = heightmap.copy()

        # Aplicar gradiente en los bordes
        edge_width = 10

        for i in range(edge_width):
            factor = i / edge_width
            result[i, :] *= factor
            result[-i-1, :] *= factor
            result[:, i] *= factor
            result[:, -i-1] *= factor

        return result

    def generate_biome_map(self) -> np.ndarray:
        """
        Genera mapa de biomas basado en altura y humedad

        Returns:
            Array 2D con tipos de bioma
        """
        if self.heightmap is None:
            self.generate_terrain()

        height, width = self.heightmap.shape
        biome_map = np.zeros((height, width), dtype=int)

        # Generar mapa de humedad
        humidity = self._generate_perlin_noise(width, height, 8)
        humidity = (humidity - humidity.min()) / (humidity.max() - humidity.min())

        # Clasificar biomas
        for y in range(height):
            for x in range(width):
                h = self.heightmap[y, x]
                hum = humidity[y, x]

                if h < self.config.sea_level:
                    biome_map[y, x] = BiomeType.OCEAN.value[0]
                elif h < self.config.sea_level + 0.05:
                    biome_map[y, x] = BiomeType.BEACH.value[0]
                elif h < 0.5:
                    if hum > 0.6:
                        biome_map[y, x] = BiomeType.FOREST.value[0]
                    else:
                        biome_map[y, x] = BiomeType.PLAINS.value[0]
                elif h < 0.7:
                    if hum > 0.7:
                        biome_map[y, x] = BiomeType.JUNGLE.value[0]
                    else:
                        biome_map[y, x] = BiomeType.FOREST.value[0]
                elif h < 0.85:
                    biome_map[y, x] = BiomeType.MOUNTAINS.value[0]
                else:
                    biome_map[y, x] = BiomeType.SNOW.value[0]

        self.biome_map = biome_map
        return biome_map

    def export_for_opensim(self) -> Dict:
        """
        Exporta terreno en formato compatible con OpenSimulator

        Returns:
            Diccionario con datos de terreno OpenSim
        """
        if self.heightmap is None:
            self.generate_terrain()

        # Escalar alturas al rango OpenSim (0-255)
        scaled_heights = (self.heightmap * 255).astype(np.uint8)

        return {
            "RegionSizeX": self.config.size_x,
            "RegionSizeY": self.config.size_y,
            "HeightMap": scaled_heights.tolist(),
            "WaterHeight": self.config.sea_level * self.config.height_scale,
            "TerrainDetail0": 1.0,
            "TerrainDetail1": 1.0,
            "TerrainDetail2": 1.0,
            "TerrainDetail3": 1.0,
            "Format": "OpenSimulator Terrain v1.0"
        }


class IslandBuilder:
    """Constructor avanzado de islas virtuales"""

    def __init__(self):
        self.terrain_gen = TerrainGenerator()
        self.features = []

    def build_island(
        self,
        size: int = 256,
        style: str = "tropical"
    ) -> Dict:
        """
        Construye una isla completa con características

        Args:
            size: Tamaño de la región (256x256 es estándar OpenSim)
            style: Estilo de isla ('tropical', 'volcanic', 'arctic')

        Returns:
            Datos completos de la isla
        """
        # Configurar generador
        config = TerrainConfig(
            size_x=size,
            size_y=size,
            seed=np.random.randint(0, 100000)
        )

        self.terrain_gen.config = config

        # Generar terreno base de isla
        island_config = IslandConfig(
            radius=size * 0.4,
            center=(size / 2, size / 2)
        )

        heightmap = self.terrain_gen.generate_island(island_config)

        # Generar biomas
        biome_map = self.terrain_gen.generate_biome_map()

        # Añadir características según estilo
        if style == "tropical":
            self._add_tropical_features(heightmap, biome_map)
        elif style == "volcanic":
            self._add_volcanic_features(heightmap)
        elif style == "arctic":
            self._add_arctic_features(heightmap)

        return {
            "heightmap": heightmap,
            "biome_map": biome_map,
            "features": self.features,
            "opensim_data": self.terrain_gen.export_for_opensim(),
            "size": size,
            "style": style
        }

    def _add_tropical_features(self, heightmap: np.ndarray, biome_map: np.ndarray):
        """Añade características tropicales"""
        # Añadir palmeras en playas
        beach_positions = np.where(biome_map == ord('b'))

        for i in range(0, len(beach_positions[0]), 20):
            y, x = beach_positions[0][i], beach_positions[1][i]
            self.features.append({
                "type": "palm_tree",
                "position": (x, heightmap[y, x], y),
                "scale": 1.0 + np.random.uniform(-0.2, 0.2)
            })

    def _add_volcanic_features(self, heightmap: np.ndarray):
        """Añade características volcánicas"""
        # Encontrar pico más alto
        max_pos = np.unravel_index(heightmap.argmax(), heightmap.shape)

        self.features.append({
            "type": "volcano_crater",
            "position": (max_pos[1], heightmap[max_pos], max_pos[0]),
            "radius": 20.0
        })

    def _add_arctic_features(self, heightmap: np.ndarray):
        """Añade características árticas"""
        # Añadir icebergs en agua
        water_positions = np.where(heightmap < 0.3)

        for i in range(0, len(water_positions[0]), 50):
            y, x = water_positions[0][i], water_positions[1][i]
            self.features.append({
                "type": "iceberg",
                "position": (x, 0.0, y),
                "scale": 1.0 + np.random.uniform(0, 2)
            })


class ProceduralLandscape:
    """Generador de paisajes procedurales completos"""

    def __init__(self):
        self.terrain_generator = TerrainGenerator()
        self.island_builder = IslandBuilder()
        self.vegetation_density = 0.5

    def generate_complete_region(
        self,
        region_type: str = "mixed",
        size: int = 256
    ) -> Dict:
        """
        Genera una región completa con terreno, vegetación y objetos

        Args:
            region_type: Tipo de región ('island', 'continent', 'mixed')
            size: Tamaño de la región

        Returns:
            Datos completos de la región
        """
        if region_type == "island":
            return self.island_builder.build_island(size)

        # Configurar terreno
        config = TerrainConfig(size_x=size, size_y=size)
        self.terrain_generator.config = config

        # Generar terreno
        heightmap = self.terrain_generator.generate_terrain()
        biome_map = self.terrain_generator.generate_biome_map()

        # Generar vegetación
        vegetation = self._generate_vegetation(heightmap, biome_map)

        # Generar puntos de interés
        poi = self._generate_points_of_interest(heightmap, biome_map)

        return {
            "type": region_type,
            "size": size,
            "heightmap": heightmap,
            "biome_map": biome_map,
            "vegetation": vegetation,
            "points_of_interest": poi,
            "opensim_data": self.terrain_generator.export_for_opensim()
        }

    def _generate_vegetation(
        self,
        heightmap: np.ndarray,
        biome_map: np.ndarray
    ) -> List[Dict]:
        """Genera vegetación según bioma"""
        vegetation = []
        height, width = heightmap.shape

        # Muestreo espaciado para vegetación
        sample_distance = int(5 / self.vegetation_density)

        for y in range(0, height, sample_distance):
            for x in range(0, width, sample_distance):
                biome = chr(biome_map[y, x])
                h = heightmap[y, x]

                # Solo generar vegetación en tierra
                if h > 0.3:
                    plant_type = self._select_plant_for_biome(biome)

                    if plant_type:
                        vegetation.append({
                            "type": plant_type,
                            "position": (x, h, y),
                            "rotation": np.random.uniform(0, 360),
                            "scale": 1.0 + np.random.uniform(-0.3, 0.3)
                        })

        return vegetation

    def _select_plant_for_biome(self, biome: str) -> Optional[str]:
        """Selecciona tipo de planta según bioma"""
        biome_plants = {
            'p': ['grass', 'flower'],  # plains
            'f': ['tree_oak', 'bush'],  # forest
            'j': ['tree_palm', 'fern'],  # jungle
            'm': ['shrub', 'rock'],  # mountains
            's': ['pine_tree']  # snow
        }

        plants = biome_plants.get(biome, [])

        if plants and np.random.random() < 0.3:
            return np.random.choice(plants)

        return None

    def _generate_points_of_interest(
        self,
        heightmap: np.ndarray,
        biome_map: np.ndarray
    ) -> List[Dict]:
        """Genera puntos de interés (estructuras, formaciones)"""
        poi = []

        # Encontrar picos altos
        threshold = np.percentile(heightmap, 90)
        peaks = np.where(heightmap > threshold)

        for i in range(min(5, len(peaks[0]))):
            y, x = peaks[0][i], peaks[1][i]
            poi.append({
                "type": "viewpoint",
                "name": f"Peak_{i+1}",
                "position": (x, heightmap[y, x], y),
                "description": "High altitude viewpoint"
            })

        # Encontrar valles
        valley_threshold = np.percentile(heightmap, 30)
        valleys = np.where((heightmap > 0.3) & (heightmap < valley_threshold))

        if len(valleys[0]) > 0:
            idx = np.random.choice(len(valleys[0]))
            y, x = valleys[0][idx], valleys[1][idx]
            poi.append({
                "type": "settlement",
                "name": "Valley Village",
                "position": (x, heightmap[y, x], y),
                "description": "Ideal location for settlement"
            })

        return poi

    def optimize_for_performance(
        self,
        region_data: Dict,
        target_quality: TerrainQuality
    ) -> Dict:
        """
        Optimiza región para rendimiento según calidad objetivo

        Args:
            region_data: Datos de la región
            target_quality: Calidad objetivo

        Returns:
            Región optimizada
        """
        quality_settings = {
            TerrainQuality.LOW: {
                "heightmap_scale": 0.25,
                "vegetation_reduction": 0.75,
                "lod_distance": 50
            },
            TerrainQuality.MEDIUM: {
                "heightmap_scale": 0.5,
                "vegetation_reduction": 0.5,
                "lod_distance": 100
            },
            TerrainQuality.HIGH: {
                "heightmap_scale": 1.0,
                "vegetation_reduction": 0.25,
                "lod_distance": 150
            },
            TerrainQuality.ULTRA: {
                "heightmap_scale": 1.0,
                "vegetation_reduction": 0.0,
                "lod_distance": 200
            }
        }

        settings = quality_settings[target_quality]

        # Reducir vegetación
        if settings["vegetation_reduction"] > 0:
            veg_count = len(region_data["vegetation"])
            keep_count = int(veg_count * (1 - settings["vegetation_reduction"]))
            region_data["vegetation"] = region_data["vegetation"][:keep_count]

        region_data["quality_settings"] = settings

        return region_data
