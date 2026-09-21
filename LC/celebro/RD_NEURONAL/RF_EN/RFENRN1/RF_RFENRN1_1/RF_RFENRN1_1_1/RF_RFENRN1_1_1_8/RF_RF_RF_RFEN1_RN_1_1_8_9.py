"""
Neurona Especializada 9: Optimización de Rendering y Performance
Sistema avanzado de shaders y optimización gráfica para metaversos
WoldVirtual3DlucIA v0.6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class RenderQuality(Enum):
    """Niveles de calidad de renderizado"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class ShaderType(Enum):
    """Tipos de shaders"""
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    GEOMETRY = "geometry"
    COMPUTE = "compute"


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento"""
    fps: float = 60.0
    frame_time: float = 16.67  # ms
    draw_calls: int = 0
    triangles: int = 0
    vertices: int = 0
    texture_memory: int = 0  # MB
    visible_objects: int = 0
    culled_objects: int = 0


@dataclass
class RenderSettings:
    """Configuración de renderizado"""
    quality: RenderQuality = RenderQuality.HIGH
    shadows_enabled: bool = True
    shadow_resolution: int = 2048
    ambient_occlusion: bool = True
    anti_aliasing: str = "MSAA"
    anisotropic_filtering: int = 16
    texture_quality: float = 1.0
    view_distance: float = 256.0
    lod_bias: float = 1.0


class ShaderManager:
    """
    Gestor de shaders y materiales
    Soporta GLSL para WebGL y OpenGL (usado en OpenSim viewers)
    """

    def __init__(self):
        self.shaders = {}
        self.active_shader = None
        self.shader_cache = {}

    def create_shader(
        self,
        name: str,
        vertex_code: str,
        fragment_code: str,
        defines: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Crea un shader program

        Args:
            name: Nombre del shader
            vertex_code: Código GLSL del vertex shader
            fragment_code: Código GLSL del fragment shader
            defines: Definiciones de preprocesador

        Returns:
            ID del shader
        """
        # Aplicar defines
        if defines:
            define_string = "\n".join([f"#define {k} {v}" for k, v in defines.items()])
            vertex_code = define_string + "\n" + vertex_code
            fragment_code = define_string + "\n" + fragment_code

        shader_id = f"shader_{len(self.shaders)}"

        self.shaders[shader_id] = {
            "name": name,
            "vertex": vertex_code,
            "fragment": fragment_code,
            "uniforms": {},
            "compiled": False
        }

        return shader_id

    def get_standard_pbr_shader(self) -> str:
        """Retorna shader PBR estándar"""
        vertex_code = """
        #version 330 core
        
        layout(location = 0) in vec3 aPosition;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec2 aTexCoord;
        layout(location = 3) in vec3 aTangent;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        out mat3 TBN;
        
        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProjection;
        uniform mat3 uNormalMatrix;
        
        void main() {
            FragPos = vec3(uModel * vec4(aPosition, 1.0));
            Normal = uNormalMatrix * aNormal;
            TexCoord = aTexCoord;
            
            // Tangent space matrix
            vec3 T = normalize(uNormalMatrix * aTangent);
            vec3 N = normalize(Normal);
            vec3 B = cross(N, T);
            TBN = mat3(T, B, N);
            
            gl_Position = uProjection * uView * vec4(FragPos, 1.0);
        }
        """

        fragment_code = """
        #version 330 core
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        in mat3 TBN;
        
        out vec4 FragColor;
        
        // Material properties
        uniform sampler2D uAlbedoMap;
        uniform sampler2D uNormalMap;
        uniform sampler2D uMetallicMap;
        uniform sampler2D uRoughnessMap;
        uniform sampler2D uAOMap;
        
        uniform vec3 uCameraPos;
        uniform vec3 uLightPositions[4];
        uniform vec3 uLightColors[4];
        
        const float PI = 3.14159265359;
        
        vec3 getNormalFromMap() {
            vec3 tangentNormal = texture(uNormalMap, TexCoord).xyz * 2.0 - 1.0;
            return normalize(TBN * tangentNormal);
        }
        
        float DistributionGGX(vec3 N, vec3 H, float roughness) {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = max(dot(N, H), 0.0);
            float NdotH2 = NdotH * NdotH;
            
            float nom = a2;
            float denom = (NdotH2 * (a2 - 1.0) + 1.0);
            denom = PI * denom * denom;
            
            return nom / denom;
        }
        
        float GeometrySchlickGGX(float NdotV, float roughness) {
            float r = (roughness + 1.0);
            float k = (r * r) / 8.0;
            
            float nom = NdotV;
            float denom = NdotV * (1.0 - k) + k;
            
            return nom / denom;
        }
        
        float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
            float NdotV = max(dot(N, V), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float ggx2 = GeometrySchlickGGX(NdotV, roughness);
            float ggx1 = GeometrySchlickGGX(NdotL, roughness);
            
            return ggx1 * ggx2;
        }
        
        vec3 fresnelSchlick(float cosTheta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
        }
        
        void main() {
            vec3 albedo = pow(texture(uAlbedoMap, TexCoord).rgb, vec3(2.2));
            float metallic = texture(uMetallicMap, TexCoord).r;
            float roughness = texture(uRoughnessMap, TexCoord).r;
            float ao = texture(uAOMap, TexCoord).r;
            
            vec3 N = getNormalFromMap();
            vec3 V = normalize(uCameraPos - FragPos);
            
            vec3 F0 = vec3(0.04);
            F0 = mix(F0, albedo, metallic);
            
            vec3 Lo = vec3(0.0);
            for(int i = 0; i < 4; ++i) {
                vec3 L = normalize(uLightPositions[i] - FragPos);
                vec3 H = normalize(V + L);
                float distance = length(uLightPositions[i] - FragPos);
                float attenuation = 1.0 / (distance * distance);
                vec3 radiance = uLightColors[i] * attenuation;
                
                float NDF = DistributionGGX(N, H, roughness);
                float G = GeometrySmith(N, V, L, roughness);
                vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                
                vec3 numerator = NDF * G * F;
                float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                vec3 specular = numerator / denominator;
                
                vec3 kS = F;
                vec3 kD = vec3(1.0) - kS;
                kD *= 1.0 - metallic;
                
                float NdotL = max(dot(N, L), 0.0);
                Lo += (kD * albedo / PI + specular) * radiance * NdotL;
            }
            
            vec3 ambient = vec3(0.03) * albedo * ao;
            vec3 color = ambient + Lo;
            
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0/2.2));
            
            FragColor = vec4(color, 1.0);
        }
        """

        return self.create_shader("PBR_Standard", vertex_code, fragment_code)

    def get_avatar_shader(self) -> str:
        """Retorna shader optimizado para avatares"""
        vertex_code = """
        #version 330 core
        
        layout(location = 0) in vec3 aPosition;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec2 aTexCoord;
        layout(location = 3) in vec4 aBoneWeights;
        layout(location = 4) in ivec4 aBoneIndices;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        
        uniform mat4 uModel;
        uniform mat4 uView;
        uniform mat4 uProjection;
        uniform mat4 uBoneMatrices[100];
        
        void main() {
            mat4 boneTransform = uBoneMatrices[aBoneIndices[0]] * aBoneWeights[0];
            boneTransform += uBoneMatrices[aBoneIndices[1]] * aBoneWeights[1];
            boneTransform += uBoneMatrices[aBoneIndices[2]] * aBoneWeights[2];
            boneTransform += uBoneMatrices[aBoneIndices[3]] * aBoneWeights[3];
            
            vec4 worldPos = uModel * boneTransform * vec4(aPosition, 1.0);
            FragPos = worldPos.xyz;
            Normal = mat3(transpose(inverse(uModel * boneTransform))) * aNormal;
            TexCoord = aTexCoord;
            
            gl_Position = uProjection * uView * worldPos;
        }
        """

        fragment_code = """
        #version 330 core
        
        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        
        out vec4 FragColor;
        
        uniform sampler2D uDiffuseMap;
        uniform sampler2D uNormalMap;
        uniform vec3 uLightDir;
        uniform vec3 uLightColor;
        
        void main() {
            vec3 albedo = texture(uDiffuseMap, TexCoord).rgb;
            vec3 normal = normalize(Normal);
            
            float diff = max(dot(normal, uLightDir), 0.0);
            vec3 diffuse = diff * uLightColor;
            
            vec3 ambient = vec3(0.3);
            
            vec3 result = (ambient + diffuse) * albedo;
            FragColor = vec4(result, 1.0);
        }
        """

        return self.create_shader("Avatar_Skinned", vertex_code, fragment_code)

    def set_uniform(self, shader_id: str, uniform_name: str, value):
        """Establece valor de uniform"""
        if shader_id in self.shaders:
            self.shaders[shader_id]["uniforms"][uniform_name] = value


class RenderingOptimizer:
    """
    Optimizador de renderizado
    Implementa técnicas de culling, batching, instancing
    """

    def __init__(self, settings: Optional[RenderSettings] = None):
        self.settings = settings or RenderSettings()
        self.metrics = PerformanceMetrics()
        self.visible_objects = []
        self.lod_system = LODSystem()
        self.frustum = None

    def optimize_scene(
        self,
        camera_position: np.ndarray,
        camera_direction: np.ndarray,
        camera_fov: float,
        objects: List[Dict]
    ) -> List[Dict]:
        """
        Optimiza escena para renderizado

        Args:
            camera_position: Posición de cámara
            camera_direction: Dirección de vista
            camera_fov: Campo de visión
            objects: Lista de objetos en la escena

        Returns:
            Lista de objetos optimizados para renderizar
        """
        # Actualizar frustum de cámara
        self.frustum = self._create_frustum(camera_position, camera_direction, camera_fov)

        # Frustum culling
        visible = self._frustum_culling(objects)

        # Occlusion culling (simplificado)
        visible = self._occlusion_culling(visible, camera_position)

        # Distance culling
        visible = self._distance_culling(visible, camera_position)

        # Seleccionar LOD apropiado
        for obj in visible:
            lod_level = self.lod_system.select_lod(
                obj.get("position", [0, 0, 0]),
                camera_position,
                obj.get("lod_distances", [10, 50, 100, 200])
            )
            obj["active_lod"] = lod_level

        # Ordenar para minimizar cambios de estado
        visible = self._sort_for_rendering(visible)

        # Actualizar métricas
        self.metrics.visible_objects = len(visible)
        self.metrics.culled_objects = len(objects) - len(visible)

        self.visible_objects = visible
        return visible

    def _create_frustum(
        self,
        position: np.ndarray,
        direction: np.ndarray,
        fov: float
    ) -> Dict:
        """Crea frustum de cámara para culling"""
        return {
            "position": position,
            "direction": direction,
            "fov": fov,
            "near": 0.1,
            "far": self.settings.view_distance
        }

    def _frustum_culling(self, objects: List[Dict]) -> List[Dict]:
        """Elimina objetos fuera del frustum de cámara"""
        if not self.frustum:
            return objects

        visible = []
        camera_pos = self.frustum["position"]
        camera_dir = self.frustum["direction"]
        fov_rad = np.radians(self.frustum["fov"])

        for obj in objects:
            obj_pos = np.array(obj.get("position", [0, 0, 0]))

            # Vector hacia objeto
            to_obj = obj_pos - camera_pos
            distance = np.linalg.norm(to_obj)

            if distance < 0.1:
                continue

            to_obj_normalized = to_obj / distance

            # Comprobar ángulo con dirección de cámara
            dot = np.dot(camera_dir, to_obj_normalized)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))

            # Visible si está dentro del FOV + margen
            if angle < fov_rad * 0.6:  # 60% del FOV como margen
                visible.append(obj)

        return visible

    def _occlusion_culling(
        self,
        objects: List[Dict],
        camera_position: np.ndarray
    ) -> List[Dict]:
        """Elimina objetos ocultos por otros (simplificado)"""
        # Implementación básica: ordenar por distancia
        # En producción usar occlusion queries u octree

        objects_with_distance = []
        for obj in objects:
            obj_pos = np.array(obj.get("position", [0, 0, 0]))
            distance = np.linalg.norm(obj_pos - camera_position)
            objects_with_distance.append((distance, obj))

        objects_with_distance.sort(key=lambda x: x[0])

        return [obj for _, obj in objects_with_distance]

    def _distance_culling(
        self,
        objects: List[Dict],
        camera_position: np.ndarray
    ) -> List[Dict]:
        """Elimina objetos demasiado lejanos"""
        max_distance = self.settings.view_distance

        visible = []
        for obj in objects:
            obj_pos = np.array(obj.get("position", [0, 0, 0]))
            distance = np.linalg.norm(obj_pos - camera_position)

            if distance <= max_distance:
                visible.append(obj)

        return visible

    def _sort_for_rendering(self, objects: List[Dict]) -> List[Dict]:
        """
        Ordena objetos para minimizar cambios de estado
        Agrupa por material y shader
        """
        def sort_key(obj):
            return (
                obj.get("shader_id", ""),
                obj.get("material_id", ""),
                obj.get("texture_id", "")
            )

        return sorted(objects, key=sort_key)

    def batch_objects(self, objects: List[Dict]) -> List[Dict]:
        """Agrupa objetos similares para batch rendering"""
        batches = {}

        for obj in objects:
            batch_key = (
                obj.get("shader_id", ""),
                obj.get("material_id", ""),
                obj.get("mesh_id", "")
            )

            if batch_key not in batches:
                batches[batch_key] = {
                    "instances": [],
                    "shader_id": obj.get("shader_id"),
                    "material_id": obj.get("material_id"),
                    "mesh_id": obj.get("mesh_id")
                }

            batches[batch_key]["instances"].append(obj)

        return list(batches.values())

    def adjust_quality_dynamic(self, target_fps: float = 60.0):
        """Ajusta calidad dinámicamente según rendimiento"""
        current_fps = self.metrics.fps

        if current_fps < target_fps * 0.8:
            # Reducir calidad
            if self.settings.quality == RenderQuality.ULTRA:
                self.settings.quality = RenderQuality.HIGH
            elif self.settings.quality == RenderQuality.HIGH:
                self.settings.quality = RenderQuality.MEDIUM
                self.settings.shadow_resolution = 1024
            elif self.settings.quality == RenderQuality.MEDIUM:
                self.settings.quality = RenderQuality.LOW
                self.settings.shadows_enabled = False

        elif current_fps > target_fps * 1.2:
            # Aumentar calidad
            if self.settings.quality == RenderQuality.LOW:
                self.settings.quality = RenderQuality.MEDIUM
                self.settings.shadows_enabled = True
            elif self.settings.quality == RenderQuality.MEDIUM:
                self.settings.quality = RenderQuality.HIGH
                self.settings.shadow_resolution = 2048
            elif self.settings.quality == RenderQuality.HIGH:
                self.settings.quality = RenderQuality.ULTRA


class LODSystem:
    """Sistema de Level of Detail"""

    def select_lod(
        self,
        object_position: Tuple[float, float, float],
        camera_position: np.ndarray,
        lod_distances: List[float]
    ) -> int:
        """
        Selecciona nivel LOD apropiado

        Args:
            object_position: Posición del objeto
            camera_position: Posición de cámara
            lod_distances: Distancias de cambio de LOD

        Returns:
            Nivel LOD (0 = máximo detalle)
        """
        obj_pos = np.array(object_position)
        distance = np.linalg.norm(obj_pos - camera_position)

        for i, lod_dist in enumerate(lod_distances):
            if distance < lod_dist:
                return i

        return len(lod_distances)


class PerformanceMonitor:
    """Monitor de rendimiento en tiempo real"""

    def __init__(self):
        self.frame_times = []
        self.max_samples = 60
        self.last_frame_time = time.time()
        self.metrics_history = []

    def begin_frame(self):
        """Marca inicio de frame"""
        self.last_frame_time = time.time()

    def end_frame(self) -> PerformanceMetrics:
        """
        Marca fin de frame y calcula métricas

        Returns:
            Métricas actualizadas
        """
        current_time = time.time()
        frame_time = (current_time - self.last_frame_time) * 1000  # ms

        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)

        avg_frame_time = np.mean(self.frame_times)
        fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 60.0

        metrics = PerformanceMetrics(
            fps=fps,
            frame_time=avg_frame_time
        )

        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)

        return metrics

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de rendimiento"""
        if not self.frame_times:
            return {}

        return {
            "avg_fps": 1000.0 / np.mean(self.frame_times),
            "min_fps": 1000.0 / np.max(self.frame_times),
            "max_fps": 1000.0 / np.min(self.frame_times),
            "avg_frame_time": np.mean(self.frame_times),
            "frame_time_std": np.std(self.frame_times)
        }
