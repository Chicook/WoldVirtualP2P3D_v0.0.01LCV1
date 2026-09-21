"""
Neurona 8: Reconocimiento y Expresiones Faciales (Facial Expression Recognition)
Algoritmos de IA y visión por computadora para detectar, generar y animar
expresiones faciales realistas en avatares 3D.

Librerías: numpy, opencv, dlib (opcional), sklearn
Técnicas: Blend Shapes, FACS (Facial Action Coding System), ML-based Expression Recognition
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV no disponible")


logger = logging.getLogger(__name__)


class FacialExpression(Enum):
    """Expresiones faciales básicas universales."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"
    CONTEMPT = "contempt"


class ActionUnit(Enum):
    """
    Action Units del sistema FACS (Facial Action Coding System).
    Representan acciones musculares individuales.
    """
    # Región superior
    AU1 = "inner_brow_raiser"
    AU2 = "outer_brow_raiser"
    AU4 = "brow_lowerer"
    AU5 = "upper_lid_raiser"
    AU6 = "cheek_raiser"
    AU7 = "lid_tightener"

    # Región inferior
    AU9 = "nose_wrinkler"
    AU10 = "upper_lip_raiser"
    AU12 = "lip_corner_puller"
    AU15 = "lip_corner_depressor"
    AU16 = "lower_lip_depressor"
    AU17 = "chin_raiser"
    AU20 = "lip_stretcher"
    AU23 = "lip_tightener"
    AU25 = "lips_part"
    AU26 = "jaw_drop"
    AU27 = "mouth_stretch"


@dataclass
class BlendShape:
    """Blend shape para animación facial."""
    name: str
    action_units: List[ActionUnit]
    vertex_deltas: np.ndarray  # Desplazamientos de vértices
    weight: float = 0.0  # 0.0 a 1.0


@dataclass
class FacialRig:
    """Rig facial completo con blend shapes."""
    base_mesh: np.ndarray  # Vértices base
    blend_shapes: List[BlendShape]
    expression_presets: Dict[FacialExpression, Dict[str, float]]


class FacialExpressionNeuron:
    """
    Neurona especializada en reconocimiento y generación de expresiones faciales
    para avatares 3D en OpenSimulator.

    Funcionalidades:
    - Generación de blend shapes faciales
    - Sistema FACS completo (Action Units)
    - Presets de expresiones emocionales
    - Animación procedural de expresiones
    - Transiciones suaves entre expresiones
    - Lip-sync automático
    - Micro-expresiones
    - Expresiones compuestas
    """

    def __init__(self):
        """Inicializa la neurona de expresiones faciales."""
        self.name = "FacialExpressionNeuron"
        self.version = "1.0.0"
        self.expression_cache = {}

        # Cargar presets de expresiones
        self.expression_presets = self._load_expression_presets()

        logger.info(f"{self.name} v{self.version} inicializada")

    def process(self, avatar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa y genera sistema de expresiones faciales para un avatar.

        Args:
            avatar_data: Diccionario con datos del avatar

        Returns:
            Diccionario con rig facial y expresiones
        """
        try:
            mesh_data = avatar_data.get('mesh', None)
            face_mesh = avatar_data.get('face_mesh', None)

            if mesh_data is None and face_mesh is None:
                return {'error': 'No se encontraron datos de malla facial'}

            # Usar face_mesh si está disponible, sino usar mesh completa
            facial_mesh = face_mesh if face_mesh is not None else mesh_data

            # Detectar landmarks faciales
            landmarks = self._detect_face_landmarks(facial_mesh)

            # Generar blend shapes basados en FACS
            blend_shapes = self._generate_blend_shapes(facial_mesh, landmarks)

            # Crear rig facial
            facial_rig = FacialRig(
                base_mesh=np.array(facial_mesh.get('vertices', [])),
                blend_shapes=blend_shapes,
                expression_presets=self.expression_presets
            )

            # Generar animaciones de muestra para cada expresión
            expression_animations = self._generate_expression_animations(facial_rig)

            # Generar sistema de lip-sync
            lipsync_system = self._generate_lipsync_system(facial_rig)

            # Generar código de animación
            animation_code = self._generate_animation_code()

            return {
                'success': True,
                'facial_rig': self._rig_to_dict(facial_rig),
                'blend_shape_count': len(blend_shapes),
                'landmarks': landmarks,
                'expression_animations': expression_animations,
                'lipsync_system': lipsync_system,
                'animation_code': animation_code,
                'supported_expressions': [e.value for e in FacialExpression]
            }

        except Exception as e:
            logger.error(f"Error en {self.name}: {e}")
            return {'error': str(e)}

    def _load_expression_presets(self) -> Dict[FacialExpression, Dict[str, float]]:
        """
        Carga presets de expresiones basados en combinaciones de Action Units.

        Returns:
            Diccionario de expresiones a pesos de blend shapes
        """
        return {
            FacialExpression.HAPPY: {
                'lip_corner_puller': 0.8,
                'cheek_raiser': 0.6,
                'outer_brow_raiser': 0.3
            },
            FacialExpression.SAD: {
                'inner_brow_raiser': 0.7,
                'lip_corner_depressor': 0.6,
                'lower_lip_depressor': 0.4
            },
            FacialExpression.ANGRY: {
                'brow_lowerer': 0.9,
                'lid_tightener': 0.6,
                'lip_tightener': 0.5,
                'jaw_drop': 0.3
            },
            FacialExpression.SURPRISED: {
                'inner_brow_raiser': 0.8,
                'outer_brow_raiser': 0.8,
                'upper_lid_raiser': 0.9,
                'jaw_drop': 0.7,
                'lips_part': 0.6
            },
            FacialExpression.DISGUSTED: {
                'nose_wrinkler': 0.8,
                'upper_lip_raiser': 0.7,
                'brow_lowerer': 0.4
            },
            FacialExpression.FEARFUL: {
                'inner_brow_raiser': 0.7,
                'outer_brow_raiser': 0.7,
                'upper_lid_raiser': 0.8,
                'lip_stretcher': 0.6,
                'lips_part': 0.5
            },
            FacialExpression.NEUTRAL: {},
            FacialExpression.CONTEMPT: {
                'lip_corner_puller': 0.5,  # Solo un lado
                'brow_lowerer': 0.3
            }
        }

    def _detect_face_landmarks(self, facial_mesh: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Detecta landmarks faciales en la malla.

        Args:
            facial_mesh: Datos de la malla facial

        Returns:
            Diccionario de landmarks detectados
        """
        vertices = np.array(facial_mesh.get('vertices', []))

        if len(vertices) == 0:
            return {}

        # Análisis geométrico para encontrar landmarks
        # Esto es una aproximación; en producción se usaría ML

        center = vertices.mean(axis=0)

        # Encontrar puntos clave por posición relativa
        landmarks = {}

        # Punto más alto = tope de la cabeza
        highest_idx = vertices[:, 1].argmax()
        landmarks['top_head'] = vertices[highest_idx]

        # Punto más bajo en la cara = barbilla
        face_vertices = vertices[vertices[:, 1] > center[1] - (vertices[:, 1].max() - center[1]) * 0.8]
        if len(face_vertices) > 0:
            lowest_idx = face_vertices[:, 1].argmin()
            landmarks['chin'] = face_vertices[lowest_idx]

        # Puntos extremos laterales = orejas/lados de cara
        leftmost_idx = vertices[:, 0].argmin()
        rightmost_idx = vertices[:, 0].argmax()
        landmarks['left_side'] = vertices[leftmost_idx]
        landmarks['right_side'] = vertices[rightmost_idx]

        # Punto más frontal = nariz
        frontmost_idx = vertices[:, 2].argmax()
        landmarks['nose_tip'] = vertices[frontmost_idx]

        return landmarks

    def _generate_blend_shapes(self, facial_mesh: Dict[str, Any],
                               landmarks: Dict[str, np.ndarray]) -> List[BlendShape]:
        """
        Genera blend shapes basados en el sistema FACS.

        Args:
            facial_mesh: Datos de la malla facial
            landmarks: Landmarks detectados

        Returns:
            Lista de blend shapes generados
        """
        vertices = np.array(facial_mesh.get('vertices', []))
        blend_shapes = []

        if len(vertices) == 0:
            return blend_shapes

        # Generar blend shapes para acciones faciales comunes
        action_units = [
            ('inner_brow_raiser', [ActionUnit.AU1]),
            ('outer_brow_raiser', [ActionUnit.AU2]),
            ('brow_lowerer', [ActionUnit.AU4]),
            ('upper_lid_raiser', [ActionUnit.AU5]),
            ('cheek_raiser', [ActionUnit.AU6]),
            ('lid_tightener', [ActionUnit.AU7]),
            ('nose_wrinkler', [ActionUnit.AU9]),
            ('upper_lip_raiser', [ActionUnit.AU10]),
            ('lip_corner_puller', [ActionUnit.AU12]),
            ('lip_corner_depressor', [ActionUnit.AU15]),
            ('lower_lip_depressor', [ActionUnit.AU16]),
            ('chin_raiser', [ActionUnit.AU17]),
            ('lip_stretcher', [ActionUnit.AU20]),
            ('lip_tightener', [ActionUnit.AU23]),
            ('lips_part', [ActionUnit.AU25]),
            ('jaw_drop', [ActionUnit.AU26]),
            ('mouth_stretch', [ActionUnit.AU27])
        ]

        for name, aus in action_units:
            # Generar desplazamientos de vértices para este blend shape
            vertex_deltas = self._generate_vertex_deltas(vertices, name, landmarks)

            blend_shape = BlendShape(
                name=name,
                action_units=aus,
                vertex_deltas=vertex_deltas,
                weight=0.0
            )
            blend_shapes.append(blend_shape)

        return blend_shapes

    def _generate_vertex_deltas(self, vertices: np.ndarray, action_name: str,
                                landmarks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Genera desplazamientos de vértices para un blend shape específico.

        Args:
            vertices: Vértices base
            action_name: Nombre de la acción facial
            landmarks: Landmarks de referencia

        Returns:
            Array de desplazamientos por vértice
        """
        deltas = np.zeros_like(vertices)

        # Región de influencia y dirección de movimiento por acción
        action_movements = {
            'inner_brow_raiser': ('eyebrow_inner', [0, 0.05, 0]),
            'outer_brow_raiser': ('eyebrow_outer', [0, 0.05, 0.01]),
            'brow_lowerer': ('eyebrow', [0, -0.03, -0.01]),
            'upper_lid_raiser': ('upper_eyelid', [0, 0.03, 0]),
            'cheek_raiser': ('cheek', [0, 0.02, 0.01]),
            'lid_tightener': ('eyelid', [0, -0.01, -0.01]),
            'nose_wrinkler': ('nose', [0, 0.01, -0.02]),
            'upper_lip_raiser': ('upper_lip', [0, 0.03, 0.01]),
            'lip_corner_puller': ('mouth_corner', [0.02, 0.02, 0]),
            'lip_corner_depressor': ('mouth_corner', [0, -0.03, 0]),
            'lower_lip_depressor': ('lower_lip', [0, -0.03, 0]),
            'chin_raiser': ('chin', [0, 0.02, 0.01]),
            'lip_stretcher': ('mouth', [0.03, 0, 0]),
            'lip_tightener': ('mouth', [-0.01, 0, 0.01]),
            'lips_part': ('lips', [0, 0.02, 0]),
            'jaw_drop': ('jaw', [0, -0.05, 0]),
            'mouth_stretch': ('mouth', [0.02, -0.02, 0])
        }

        if action_name not in action_movements:
            return deltas

        region, movement = action_movements[action_name]
        movement = np.array(movement)

        # Aplicar movimiento a vértices en la región afectada
        # Esto es una aproximación; en producción se usarían weight maps
        center = vertices.mean(axis=0)

        for i, vertex in enumerate(vertices):
            # Calcular si el vértice está en la región de influencia
            influence = self._calculate_region_influence(vertex, region, center, landmarks)

            if influence > 0:
                deltas[i] = movement * influence

        return deltas

    def _calculate_region_influence(self, vertex: np.ndarray, region: str,
                                    center: np.ndarray,
                                    landmarks: Dict[str, np.ndarray]) -> float:
        """
        Calcula la influencia de una región facial sobre un vértice.

        Args:
            vertex: Vértice a evaluar
            region: Nombre de la región facial
            center: Centro de la cara
            landmarks: Landmarks de referencia

        Returns:
            Factor de influencia [0, 1]
        """
        # Mapeo de regiones a posiciones relativas
        region_positions = {
            'eyebrow': (0, 0.3, 0),
            'eyebrow_inner': (-0.05, 0.3, 0.05),
            'eyebrow_outer': (0.15, 0.3, 0.05),
            'upper_eyelid': (0, 0.25, 0.1),
            'eyelid': (0, 0.25, 0.08),
            'cheek': (0.1, 0.1, 0.05),
            'nose': (0, 0.15, 0.15),
            'upper_lip': (0, -0.05, 0.12),
            'lower_lip': (0, -0.1, 0.12),
            'mouth': (0, -0.08, 0.1),
            'mouth_corner': (0.08, -0.05, 0.1),
            'lips': (0, -0.08, 0.12),
            'jaw': (0, -0.25, 0.05),
            'chin': (0, -0.3, 0.1)
        }

        if region not in region_positions:
            return 0.0

        # Posición de la región relativa al centro
        region_pos = center + np.array(region_positions[region])

        # Calcular distancia del vértice a la región
        distance = np.linalg.norm(vertex - region_pos)

        # Radio de influencia (ajustable)
        influence_radius = 0.1

        # Calcular influencia con falloff suave
        if distance < influence_radius:
            influence = 1.0 - (distance / influence_radius) ** 2
            return float(influence)

        return 0.0

    def _generate_expression_animations(self, facial_rig: FacialRig) -> List[Dict[str, Any]]:
        """
        Genera animaciones de muestra para cada expresión.

        Args:
            facial_rig: Rig facial

        Returns:
            Lista de animaciones por expresión
        """
        animations = []

        for expression, weights in facial_rig.expression_presets.items():
            # Generar keyframes de transición
            keyframes = []

            # Frame 0: neutral
            keyframes.append({
                'time': 0.0,
                'weights': {bs.name: 0.0 for bs in facial_rig.blend_shapes}
            })

            # Frame 30: expresión completa (0.5 segundos a 60 FPS)
            keyframes.append({
                'time': 0.5,
                'weights': weights
            })

            # Frame 60: vuelta a neutral
            keyframes.append({
                'time': 1.0,
                'weights': {bs.name: 0.0 for bs in facial_rig.blend_shapes}
            })

            animations.append({
                'expression': expression.value,
                'duration': 1.0,
                'keyframes': keyframes,
                'loop': False
            })

        return animations

    def _generate_lipsync_system(self, facial_rig: FacialRig) -> Dict[str, Any]:
        """
        Genera sistema de lip-sync para el rig facial.

        Args:
            facial_rig: Rig facial

        Returns:
            Configuración del sistema de lip-sync
        """
        # Visemas básicos para lip-sync
        visemes = {
            'silence': {},
            'PP': {'lip_tightener': 1.0},  # P, B, M
            'FF': {'lower_lip_depressor': 0.6, 'upper_lip_raiser': 0.3},  # F, V
            'TH': {'lips_part': 0.3},  # TH
            'DD': {'jaw_drop': 0.3, 'lips_part': 0.4},  # D, T, N
            'kk': {'jaw_drop': 0.4, 'mouth_stretch': 0.3},  # K, G
            'CH': {'lip_corner_puller': 0.3, 'lips_part': 0.5},  # CH, J, SH
            'SS': {'lip_stretcher': 0.5, 'lips_part': 0.2},  # S, Z
            'nn': {'jaw_drop': 0.2},  # NN
            'RR': {'lip_corner_puller': 0.2, 'lips_part': 0.3},  # R
            'aa': {'jaw_drop': 0.8, 'lips_part': 0.7},  # AA (father)
            'E': {'lip_stretcher': 0.6, 'jaw_drop': 0.3},  # E (bed)
            'I': {'lip_stretcher': 0.7, 'jaw_drop': 0.2},  # I (treat)
            'O': {'lips_part': 0.6, 'jaw_drop': 0.4},  # O (boat)
            'U': {'lip_tightener': 0.6, 'lips_part': 0.5}  # U (boot)
        }

        return {
            'visemes': visemes,
            'transition_time': 0.1,  # Tiempo de transición entre visemas
            'phoneme_mapping': {
                'a': 'aa', 'e': 'E', 'i': 'I', 'o': 'O', 'u': 'U',
                'p': 'PP', 'b': 'PP', 'm': 'PP',
                'f': 'FF', 'v': 'FF',
                't': 'DD', 'd': 'DD', 'n': 'DD',
                'k': 'kk', 'g': 'kk',
                's': 'SS', 'z': 'SS',
                'r': 'RR'
            }
        }

    def _generate_animation_code(self) -> Dict[str, str]:
        """Genera código para sistema de animación facial."""
        python_code = """
# Sistema de Animación Facial
class FacialAnimator:
    def __init__(self, facial_rig):
        self.rig = facial_rig
        self.current_weights = {bs.name: 0.0 for bs in facial_rig.blend_shapes}
    
    def apply_expression(self, expression_name, intensity=1.0):
        '''Aplica una expresión con intensidad controlada'''
        if expression_name not in self.rig.expression_presets:
            return
        
        target_weights = self.rig.expression_presets[expression_name]
        
        for bs_name, weight in target_weights.items():
            self.current_weights[bs_name] = weight * intensity
    
    def blend_expressions(self, expressions_weights):
        '''Mezcla múltiples expresiones'''
        blended = {}
        
        for expr, expr_weight in expressions_weights.items():
            if expr not in self.rig.expression_presets:
                continue
            
            for bs_name, bs_weight in self.rig.expression_presets[expr].items():
                if bs_name not in blended:
                    blended[bs_name] = 0.0
                blended[bs_name] += bs_weight * expr_weight
        
        # Normalizar
        for bs_name in blended:
            blended[bs_name] = min(blended[bs_name], 1.0)
        
        return blended
    
    def update_mesh(self, dt):
        '''Actualiza la geometría de la malla con los blend shapes actuales'''
        result_vertices = self.rig.base_mesh.copy()
        
        for blend_shape in self.rig.blend_shapes:
            weight = self.current_weights.get(blend_shape.name, 0.0)
            if weight > 0:
                result_vertices += blend_shape.vertex_deltas * weight
        
        return result_vertices
"""

        return {
            'language': 'python',
            'code': python_code,
            'description': 'Sistema de animación facial con blend shapes'
        }

    def _rig_to_dict(self, facial_rig: FacialRig) -> Dict[str, Any]:
        """Convierte un FacialRig a diccionario."""
        return {
            'base_mesh_vertices': len(facial_rig.base_mesh),
            'blend_shapes': [{
                'name': bs.name,
                'action_units': [au.value for au in bs.action_units],
                'vertex_count': len(bs.vertex_deltas)
            } for bs in facial_rig.blend_shapes],
            'expression_presets': {
                expr.value: weights
                for expr, weights in facial_rig.expression_presets.items()
            }
        }
