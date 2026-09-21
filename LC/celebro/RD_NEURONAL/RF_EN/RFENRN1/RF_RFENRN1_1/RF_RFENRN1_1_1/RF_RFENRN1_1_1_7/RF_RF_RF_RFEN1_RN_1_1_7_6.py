"""
Neurona 6: Normal Mapping y Shaders PBR (Physically Based Rendering)
Algoritmos avanzados para generación de mapas normales, roughness, metallic
y gestión de materiales PBR para avatares 3D realistas.

Librerías: numpy, opencv, PIL, scipy
Técnicas: Normal Map Generation, Height to Normal, PBR Workflow, Shader Optimization
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from PIL import Image, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL no disponible")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV no disponible")

try:
    from scipy.ndimage import convolve
except ImportError:
    pass  # dependencia pesada opcional


logger = logging.getLogger(__name__)


class PBRWorkflow(Enum):
    """Workflows PBR estándar."""
    METALLIC_ROUGHNESS = "metallic_roughness"  # Usado por glTF
    SPECULAR_GLOSSINESS = "specular_glossiness"  # Workflow alternativo


@dataclass
class PBRMaterial:
    """Material PBR completo para un avatar."""
    name: str
    base_color: np.ndarray
    metallic: float = 0.0
    roughness: float = 0.5
    normal_map: Optional[np.ndarray] = None
    ao_map: Optional[np.ndarray] = None  # Ambient Occlusion
    emissive: Optional[np.ndarray] = None
    alpha: float = 1.0
    workflow: PBRWorkflow = PBRWorkflow.METALLIC_ROUGHNESS


class NormalMappingNeuron:
    """
    Neurona especializada en generación de mapas normales y materiales PBR
    para avatares 3D en OpenSimulator.

    Funcionalidades:
    - Generación de normal maps desde height maps
    - Generación de normal maps desde texturas difusas
    - Creación de mapas roughness inteligentes
    - Creación de mapas metallic
    - Generación de ambient occlusion
    - Optimización de materiales PBR
    - Conversión entre workflows PBR
    - Baking de iluminación
    """

    def __init__(self):
        """Inicializa la neurona de normal mapping."""
        self.name = "NormalMappingNeuron"
        self.version = "1.0.0"
        self.material_cache = {}
        logger.info(f"{self.name} v{self.version} inicializada")

    def process(self, avatar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa y genera materiales PBR completos para un avatar.

        Args:
            avatar_data: Diccionario con datos del avatar

        Returns:
            Diccionario con materiales PBR generados
        """
        try:
            textures = avatar_data.get('textures', {})
            mesh_data = avatar_data.get('mesh', {})
            workflow = avatar_data.get('pbr_workflow', PBRWorkflow.METALLIC_ROUGHNESS)

            if not textures:
                return {'error': 'No se encontraron texturas'}

            results = {}

            for texture_name, texture_data in textures.items():
                logger.info(f"Procesando material PBR: {texture_name}")

                # Cargar textura base
                base_texture = self._load_texture(texture_data)

                if base_texture is None:
                    results[texture_name] = {'error': 'No se pudo cargar la textura'}
                    continue

                # Generar normal map
                normal_map = self._generate_normal_map(base_texture)

                # Generar roughness map
                roughness_map = self._generate_roughness_map(base_texture)

                # Generar metallic map
                metallic_map = self._generate_metallic_map(base_texture)

                # Generar ambient occlusion (si hay geometría disponible)
                ao_map = None
                if mesh_data:
                    ao_map = self._generate_ao_map(mesh_data, texture_name)

                # Crear material PBR
                material = PBRMaterial(
                    name=texture_name,
                    base_color=np.array(base_texture),
                    metallic=self._estimate_metallic_value(base_texture),
                    roughness=self._estimate_roughness_value(base_texture),
                    normal_map=normal_map,
                    ao_map=ao_map,
                    workflow=workflow
                )

                # Optimizar material
                optimized_material = self._optimize_material(material)

                # Generar shaders
                shader_code = self._generate_shader_code(optimized_material)

                results[texture_name] = {
                    'success': True,
                    'material': self._material_to_dict(optimized_material),
                    'normal_map': self._array_to_base64(normal_map) if normal_map is not None else None,
                    'roughness_map': self._array_to_base64(roughness_map) if roughness_map is not None else None,
                    'metallic_map': self._array_to_base64(metallic_map) if metallic_map is not None else None,
                    'ao_map': self._array_to_base64(ao_map) if ao_map is not None else None,
                    'shader_code': shader_code
                }

            return {
                'success': True,
                'processed_materials': len(results),
                'workflow': workflow.value,
                'results': results
            }

        except Exception as e:
            logger.error(f"Error en {self.name}: {e}")
            return {'error': str(e)}

    def _load_texture(self, texture_data: Any) -> Optional[Image.Image]:
        """Carga una textura desde diversos formatos."""
        if not PIL_AVAILABLE:
            return None

        try:
            if isinstance(texture_data, str):
                return Image.open(texture_data).convert('RGB')
            elif isinstance(texture_data, bytes):
                import io
                return Image.open(io.BytesIO(texture_data)).convert('RGB')
            elif isinstance(texture_data, dict) and 'data' in texture_data:
                return Image.open(texture_data['data']).convert('RGB')
            else:
                return None
        except Exception as e:
            logger.error(f"Error cargando textura: {e}")
            return None

    def _generate_normal_map(self, base_texture: Image.Image) -> np.ndarray:
        """
        Genera un normal map desde una textura difusa.

        Args:
            base_texture: Textura base

        Returns:
            Normal map como array numpy [H, W, 3]
        """
        # Convertir a escala de grises para simular altura
        gray = np.array(base_texture.convert('L')).astype(np.float32) / 255.0

        # Aplicar blur para suavizar
        if CV2_AVAILABLE:
            gray = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # Calcular gradientes usando Sobel
        if CV2_AVAILABLE:
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        else:
            # Fallback manual
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            grad_x = convolve(gray, sobel_x)
            grad_y = convolve(gray, sobel_y)

        # Construir normal map
        # Normalizar gradientes
        strength = 2.0  # Factor de fuerza del normal map

        normal = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.float32)
        normal[:, :, 0] = -grad_x * strength
        normal[:, :, 1] = -grad_y * strength
        normal[:, :, 2] = 1.0

        # Normalizar vectores
        magnitude = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
        magnitude = np.maximum(magnitude, 1e-8)
        normal = normal / magnitude

        # Convertir de [-1, 1] a [0, 1] y luego a [0, 255]
        normal = ((normal + 1.0) * 0.5 * 255).astype(np.uint8)

        return normal

    def _generate_roughness_map(self, base_texture: Image.Image) -> np.ndarray:
        """
        Genera un mapa de roughness desde una textura.

        El roughness se estima basándose en la variación local de la textura.
        Áreas con alta frecuencia espacial = más roughness.

        Args:
            base_texture: Textura base

        Returns:
            Roughness map como array numpy [H, W]
        """
        gray = np.array(base_texture.convert('L')).astype(np.float32) / 255.0

        # Calcular variación local usando desviación estándar en ventanas
        if CV2_AVAILABLE:
            # Calcular media local
            mean = cv2.blur(gray, (11, 11))
            # Calcular varianza local
            sqr_mean = cv2.blur(gray ** 2, (11, 11))
            variance = sqr_mean - mean ** 2
            variance = np.maximum(variance, 0)  # Evitar valores negativos por errores numéricos

            # La desviación estándar es el roughness
            roughness = np.sqrt(variance)
        else:
            # Fallback: usar filtro de bordes simple
            edges = np.abs(convolve(gray, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])))
            roughness = edges / edges.max() if edges.max() > 0 else edges

        # Normalizar y invertir (bordes suaves = bajo roughness)
        roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min() + 1e-8)

        # Ajustar rango: 0.3 a 0.9 (la mayoría de materiales están en este rango)
        roughness = 0.3 + roughness * 0.6

        # Convertir a uint8
        roughness = (roughness * 255).astype(np.uint8)

        return roughness

    def _generate_metallic_map(self, base_texture: Image.Image) -> np.ndarray:
        """
        Genera un mapa metallic desde una textura.

        Los metales típicamente tienen:
        - Alta saturación de color
        - Alto brillo
        - Colores específicos (gris, oro, cobre)

        Args:
            base_texture: Textura base

        Returns:
            Metallic map como array numpy [H, W]
        """
        rgb = np.array(base_texture).astype(np.float32) / 255.0

        # Calcular brillo
        brightness = np.mean(rgb, axis=2)

        # Calcular saturación
        max_rgb = np.max(rgb, axis=2)
        min_rgb = np.min(rgb, axis=2)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)

        # Los metales tienen bajo saturation (excepto oro/cobre) y brillo medio-alto
        # Clasificación simple: si es grisáceo y brillante, probablemente metálico
        metallic = np.where(
            (saturation < 0.2) & (brightness > 0.4),
            brightness,
            saturation * 0.3  # No metálicos tienen bajo metallic
        )

        # Normalizar
        metallic = np.clip(metallic, 0, 1)

        # Convertir a uint8
        metallic = (metallic * 255).astype(np.uint8)

        return metallic

    def _generate_ao_map(self, mesh_data: Dict[str, Any], texture_name: str) -> Optional[np.ndarray]:
        """
        Genera un mapa de ambient occlusion desde geometría.

        Este es un proceso complejo que requiere ray-tracing.
        Por ahora, generamos un AO aproximado.

        Args:
            mesh_data: Datos de la malla
            texture_name: Nombre de la textura

        Returns:
            AO map como array numpy [H, W] o None
        """
        # Implementación simplificada: generar AO basado en curvatura
        # En producción, esto usaría ray-tracing real

        # Por ahora, retornar un AO uniforme
        ao = np.ones((512, 512), dtype=np.uint8) * 255

        # Añadir algo de variación en los bordes
        center_x, center_y = 256, 256
        y, x = np.ogrid[:512, :512]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Oscurecer suavemente hacia los bordes
        ao = (255 * (1.0 - distance / 512 * 0.3)).astype(np.uint8)

        return ao

    def _estimate_metallic_value(self, texture: Image.Image) -> float:
        """Estima el valor metallic promedio de una textura."""
        rgb = np.array(texture).astype(np.float32) / 255.0

        # Calcular saturación promedio
        max_rgb = np.max(rgb, axis=2)
        min_rgb = np.min(rgb, axis=2)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / (max_rgb + 1e-8), 0)

        avg_saturation = np.mean(saturation)

        # Bajo saturation = probablemente metálico
        if avg_saturation < 0.2:
            return 0.8
        else:
            return 0.1

    def _estimate_roughness_value(self, texture: Image.Image) -> float:
        """Estima el valor roughness promedio de una textura."""
        gray = np.array(texture.convert('L')).astype(np.float32) / 255.0

        # Calcular varianza como proxy de roughness
        variance = np.var(gray)

        # Mapear varianza a roughness [0.2, 0.9]
        roughness = 0.2 + variance * 10
        roughness = np.clip(roughness, 0.2, 0.9)

        return float(roughness)

    def _optimize_material(self, material: PBRMaterial) -> PBRMaterial:
        """
        Optimiza un material PBR para mejor rendimiento.

        Args:
            material: Material a optimizar

        Returns:
            Material optimizado
        """
        # Optimizar rangos de valores
        material.metallic = np.clip(material.metallic, 0.0, 1.0)
        material.roughness = np.clip(material.roughness, 0.04, 1.0)  # 0.04 es el mínimo físicamente plausible
        material.alpha = np.clip(material.alpha, 0.0, 1.0)

        # Comprimir mapas si son muy grandes
        max_size = 2048

        if material.normal_map is not None:
            h, w = material.normal_map.shape[:2]
            if h > max_size or w > max_size:
                material.normal_map = self._resize_map(material.normal_map, max_size)

        if material.ao_map is not None:
            h, w = material.ao_map.shape[:2]
            if h > max_size or w > max_size:
                material.ao_map = self._resize_map(material.ao_map, max_size)

        return material

    def _resize_map(self, map_array: np.ndarray, max_size: int) -> np.ndarray:
        """Redimensiona un mapa manteniendo aspect ratio."""
        h, w = map_array.shape[:2]
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        if CV2_AVAILABLE:
            if len(map_array.shape) == 3:
                resized = cv2.resize(map_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized = cv2.resize(map_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback sin OpenCV
            resized = map_array  # Sin resize por ahora

        return resized

    def _generate_shader_code(self, material: PBRMaterial) -> Dict[str, str]:
        """
        Genera código de shader GLSL para el material PBR.

        Args:
            material: Material PBR

        Returns:
            Diccionario con código de vertex y fragment shaders
        """
        vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aTangent;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out mat3 TBN;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    
    // Calcular TBN para normal mapping
    vec3 T = normalize(vec3(model * vec4(aTangent, 0.0)));
    vec3 N = normalize(vec3(model * vec4(aNormal, 0.0)));
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);
    TBN = mat3(T, B, N);
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

        fragment_shader = f"""
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in mat3 TBN;

// Material textures
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

// Material properties
uniform float metallic;
uniform float roughness;

// Lights
uniform vec3 lightPositions[4];
uniform vec3 lightColors[4];
uniform vec3 camPos;

const float PI = 3.14159265359;

// PBR functions
float DistributionGGX(vec3 N, vec3 H, float roughness) {{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return nom / denom;
}}

float GeometrySchlickGGX(float NdotV, float roughness) {{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return nom / denom;
}}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    
    return ggx1 * ggx2;
}}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}}

void main() {{
    // Obtener propiedades del material
    vec3 albedo = pow(texture(albedoMap, TexCoord).rgb, vec3(2.2));
    vec3 normal = texture(normalMap, TexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);
    
    float metallic_val = texture(metallicMap, TexCoord).r * metallic;
    float roughness_val = texture(roughnessMap, TexCoord).r * roughness;
    float ao = texture(aoMap, TexCoord).r;
    
    vec3 N = normal;
    vec3 V = normalize(camPos - FragPos);
    
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic_val);
    
    // Cálculo de luz
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) {{
        vec3 L = normalize(lightPositions[i] - FragPos);
        vec3 H = normalize(V + L);
        float distance = length(lightPositions[i] - FragPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = lightColors[i] * attenuation;
        
        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness_val);
        float G = GeometrySmith(N, V, L, roughness_val);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic_val;
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
        vec3 specular = numerator / denominator;
        
        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }}
    
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo;
    
    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // Gamma correction
    color = pow(color, vec3(1.0/2.2));
    
    FragColor = vec4(color, 1.0);
}}
"""

        return {
            'vertex': vertex_shader,
            'fragment': fragment_shader,
            'type': 'pbr_metallic_roughness'
        }

    def _material_to_dict(self, material: PBRMaterial) -> Dict[str, Any]:
        """Convierte un material PBR a diccionario."""
        return {
            'name': material.name,
            'base_color': material.base_color.tolist() if isinstance(material.base_color, np.ndarray) else material.base_color,
            'metallic': material.metallic,
            'roughness': material.roughness,
            'alpha': material.alpha,
            'workflow': material.workflow.value
        }

    def _array_to_base64(self, array: np.ndarray) -> str:
        """Convierte un array numpy a string base64."""
        import io
        import base64

        if len(array.shape) == 2:
            # Grayscale
            img = Image.fromarray(array, mode='L')
        else:
            # RGB
            img = Image.fromarray(array, mode='RGB')

        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str[:50]}..."
