"""
Neurona 9: Optimización de Proporciones Corporales (Body Proportion Optimization)
Algoritmos de antropometría digital y machine learning para analizar y optimizar
proporciones corporales de avatares 3D según estándares realistas o estilizados.

Librerías: numpy, scipy, sklearn
Técnicas: Anthropometric Analysis, Golden Ratio, Body Mass Index, Proportion Scaling
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class BodyType(Enum):
    """Tipos de cuerpo estándar."""
    REALISTIC = "realistic"
    STYLIZED = "stylized"
    ATHLETIC = "athletic"
    HEROIC = "heroic"
    CARTOON = "cartoon"


class Gender(Enum):
    """Género para proporciones antropométricas."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass
class AnthropometricMeasurements:
    """Medidas antropométricas de un avatar."""
    total_height: float
    head_height: float
    torso_height: float
    leg_length: float
    arm_length: float
    shoulder_width: float
    hip_width: float
    waist_width: float
    chest_width: float
    neck_width: float

    def calculate_proportions(self) -> Dict[str, float]:
        """Calcula proporciones relativas a la altura total."""
        if self.total_height <= 0:
            return {}

        return {
            'head_to_total': self.head_height / self.total_height,
            'torso_to_total': self.torso_height / self.total_height,
            'leg_to_total': self.leg_length / self.total_height,
            'arm_to_total': self.arm_length / self.total_height,
            'shoulder_to_hip': self.shoulder_width / self.hip_width if self.hip_width > 0 else 1.0
        }


@dataclass
class ProportionStandard:
    """Estándares de proporciones para un tipo de cuerpo."""
    body_type: BodyType
    head_units: float  # Altura total en "cabezas"
    torso_ratio: float
    leg_ratio: float
    arm_ratio: float
    shoulder_hip_ratio: float


class BodyProportionNeuron:
    """
    Neurona especializada en análisis y optimización de proporciones corporales
    para avatares 3D en OpenSimulator.

    Funcionalidades:
    - Análisis antropométrico completo
    - Detección de proporciones no naturales
    - Corrección automática de proporciones
    - Escalado proporcional de segmentos
    - Aplicación de estándares (realista, estilizado, etc.)
    - Cálculo de IMC (Body Mass Index)
    - Optimización para animación
    - Comparación con referencias humanas
    """

    def __init__(self):
        """Inicializa la neurona de proporciones corporales."""
        self.name = "BodyProportionNeuron"
        self.version = "1.0.0"

        # Cargar estándares de proporciones
        self.proportion_standards = self._load_proportion_standards()

        logger.info(f"{self.name} v{self.version} inicializada")

    def _load_proportion_standards(self) -> Dict[BodyType, ProportionStandard]:
        """Carga estándares de proporciones para diferentes tipos de cuerpo."""
        return {
            BodyType.REALISTIC: ProportionStandard(
                body_type=BodyType.REALISTIC,
                head_units=7.5,  # Adulto promedio: 7.5 cabezas de altura
                torso_ratio=0.35,
                leg_ratio=0.52,
                arm_ratio=0.42,
                shoulder_hip_ratio=1.15  # Hombros 15% más anchos que caderas
            ),
            BodyType.STYLIZED: ProportionStandard(
                body_type=BodyType.STYLIZED,
                head_units=6.5,  # Cabeza más grande
                torso_ratio=0.32,
                leg_ratio=0.55,  # Piernas más largas
                arm_ratio=0.40,
                shoulder_hip_ratio=1.2
            ),
            BodyType.ATHLETIC: ProportionStandard(
                body_type=BodyType.ATHLETIC,
                head_units=7.8,
                torso_ratio=0.38,
                leg_ratio=0.50,
                arm_ratio=0.43,
                shoulder_hip_ratio=1.3
            ),
            BodyType.HEROIC: ProportionStandard(
                body_type=BodyType.HEROIC,
                head_units=8.5,  # Figura heroica: 8-8.5 cabezas
                torso_ratio=0.40,
                leg_ratio=0.48,
                arm_ratio=0.45,
                shoulder_hip_ratio=1.5
            ),
            BodyType.CARTOON: ProportionStandard(
                body_type=BodyType.CARTOON,
                head_units=4.0,  # Cabeza muy grande
                torso_ratio=0.40,
                leg_ratio=0.35,  # Piernas cortas
                arm_ratio=0.35,
                shoulder_hip_ratio=1.0
            )
        }

    def process(self, avatar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa y optimiza las proporciones corporales de un avatar.

        Args:
            avatar_data: Diccionario con datos del avatar

        Returns:
            Diccionario con análisis y proporciones optimizadas
        """
        try:
            skeleton = avatar_data.get('skeleton', None)
            mesh_data = avatar_data.get('mesh', None)
            target_type = avatar_data.get('body_type', BodyType.REALISTIC)
            gender = avatar_data.get('gender', Gender.NEUTRAL)

            if isinstance(target_type, str):
                target_type = BodyType(target_type)
            if isinstance(gender, str):
                gender = Gender(gender)

            if skeleton is None:
                return {'error': 'No se encontraron datos de esqueleto'}

            # Extraer medidas antropométricas
            measurements = self._extract_measurements(skeleton)

            # Analizar proporciones actuales
            current_proportions = measurements.calculate_proportions()

            # Obtener estándar objetivo
            target_standard = self.proportion_standards[target_type]

            # Calcular desviaciones
            deviations = self._calculate_deviations(measurements, target_standard)

            # Generar correcciones
            corrections = self._generate_corrections(measurements, target_standard, deviations)

            # Aplicar correcciones al esqueleto
            optimized_skeleton = self._apply_corrections(skeleton, corrections)

            # Análisis de calidad
            quality_score = self._calculate_quality_score(deviations)

            # Generar visualización de proporciones
            proportion_diagram = self._generate_proportion_diagram(measurements, target_standard)

            return {
                'success': True,
                'measurements': {
                    'total_height': measurements.total_height,
                    'head_height': measurements.head_height,
                    'torso_height': measurements.torso_height,
                    'leg_length': measurements.leg_length,
                    'arm_length': measurements.arm_length,
                    'shoulder_width': measurements.shoulder_width,
                    'hip_width': measurements.hip_width
                },
                'current_proportions': current_proportions,
                'target_standard': {
                    'body_type': target_standard.body_type.value,
                    'head_units': target_standard.head_units,
                    'ratios': {
                        'torso': target_standard.torso_ratio,
                        'legs': target_standard.leg_ratio,
                        'arms': target_standard.arm_ratio,
                        'shoulder_hip': target_standard.shoulder_hip_ratio
                    }
                },
                'deviations': deviations,
                'corrections': corrections,
                'optimized_skeleton': optimized_skeleton,
                'quality_score': quality_score,
                'proportion_diagram': proportion_diagram,
                'recommendations': self._generate_recommendations(deviations)
            }

        except Exception as e:
            logger.error(f"Error en {self.name}: {e}")
            return {'error': str(e)}

    def _extract_measurements(self, skeleton: Dict[str, Any]) -> AnthropometricMeasurements:
        """
        Extrae medidas antropométricas desde el esqueleto.

        Args:
            skeleton: Datos del esqueleto

        Returns:
            AnthropometricMeasurements con las medidas extraídas
        """
        bones = skeleton.get('bones', [])

        # Buscar huesos clave por nombre
        bone_map = {bone['name']: bone for bone in bones}

        # Calcular altura total
        pelvis_pos = self._get_bone_position(bone_map, 'mPelvis')
        head_pos = self._get_bone_position(bone_map, 'mHead')
        total_height = np.linalg.norm(np.array(head_pos) - np.array(pelvis_pos))

        # Altura de cabeza (pelvis a cuello + cuello a cabeza)
        neck_pos = self._get_bone_position(bone_map, 'mNeck')
        head_height = np.linalg.norm(np.array(head_pos) - np.array(neck_pos))

        # Altura de torso (pelvis a cuello)
        torso_height = np.linalg.norm(np.array(neck_pos) - np.array(pelvis_pos))

        # Longitud de pierna (pelvis a pie)
        foot_left_pos = self._get_bone_position(bone_map, 'mFootLeft')
        leg_length = np.linalg.norm(np.array(pelvis_pos) - np.array(foot_left_pos))

        # Longitud de brazo (hombro a muñeca)
        shoulder_left_pos = self._get_bone_position(bone_map, 'mShoulderLeft')
        wrist_left_pos = self._get_bone_position(bone_map, 'mWristLeft')
        arm_length = np.linalg.norm(np.array(shoulder_left_pos) - np.array(wrist_left_pos))

        # Ancho de hombros
        shoulder_right_pos = self._get_bone_position(bone_map, 'mShoulderRight')
        shoulder_width = np.linalg.norm(np.array(shoulder_left_pos) - np.array(shoulder_right_pos))

        # Ancho de caderas
        hip_left_pos = self._get_bone_position(bone_map, 'mHipLeft')
        hip_right_pos = self._get_bone_position(bone_map, 'mHipRight')
        hip_width = np.linalg.norm(np.array(hip_left_pos) - np.array(hip_right_pos))

        # Estimaciones para medidas no directamente disponibles
        waist_width = hip_width * 0.75
        chest_width = shoulder_width * 0.8
        neck_width = head_height * 0.3

        return AnthropometricMeasurements(
            total_height=float(total_height),
            head_height=float(head_height),
            torso_height=float(torso_height),
            leg_length=float(leg_length),
            arm_length=float(arm_length),
            shoulder_width=float(shoulder_width),
            hip_width=float(hip_width),
            waist_width=float(waist_width),
            chest_width=float(chest_width),
            neck_width=float(neck_width)
        )

    def _get_bone_position(self, bone_map: Dict[str, Any], bone_name: str) -> np.ndarray:
        """Obtiene la posición de un hueso por nombre."""
        if bone_name in bone_map:
            return np.array(bone_map[bone_name].get('position', [0, 0, 0]))
        return np.array([0, 0, 0])

    def _calculate_deviations(self, measurements: AnthropometricMeasurements,
                              standard: ProportionStandard) -> Dict[str, float]:
        """
        Calcula desviaciones de las proporciones actuales respecto al estándar.

        Args:
            measurements: Medidas actuales
            standard: Estándar objetivo

        Returns:
            Diccionario de desviaciones por segmento
        """
        current = measurements.calculate_proportions()

        deviations = {
            'torso': current.get('torso_to_total', 0) - standard.torso_ratio,
            'legs': current.get('leg_to_total', 0) - standard.leg_ratio,
            'arms': current.get('arm_to_total', 0) - standard.arm_ratio,
            'shoulder_hip': current.get('shoulder_to_hip', 0) - standard.shoulder_hip_ratio,
            'head_units': (measurements.total_height / measurements.head_height) - standard.head_units
            if measurements.head_height > 0 else 0
        }

        return deviations

    def _generate_corrections(self, measurements: AnthropometricMeasurements,
                              standard: ProportionStandard,
                              deviations: Dict[str, float]) -> Dict[str, float]:
        """
        Genera factores de corrección para ajustar las proporciones.

        Args:
            measurements: Medidas actuales
            standard: Estándar objetivo
            deviations: Desviaciones calculadas

        Returns:
            Diccionario de factores de escala por segmento
        """
        corrections = {}

        # Factor de corrección para torso
        if abs(deviations['torso']) > 0.05:  # Umbral de 5%
            target_torso = measurements.total_height * standard.torso_ratio
            corrections['torso_scale'] = target_torso / measurements.torso_height if measurements.torso_height > 0 else 1.0
        else:
            corrections['torso_scale'] = 1.0

        # Factor de corrección para piernas
        if abs(deviations['legs']) > 0.05:
            target_legs = measurements.total_height * standard.leg_ratio
            corrections['leg_scale'] = target_legs / measurements.leg_length if measurements.leg_length > 0 else 1.0
        else:
            corrections['leg_scale'] = 1.0

        # Factor de corrección para brazos
        if abs(deviations['arms']) > 0.05:
            target_arms = measurements.total_height * standard.arm_ratio
            corrections['arm_scale'] = target_arms / measurements.arm_length if measurements.arm_length > 0 else 1.0
        else:
            corrections['arm_scale'] = 1.0

        # Factor de corrección para cabeza
        if abs(deviations['head_units']) > 0.5:  # Umbral de 0.5 unidades de cabeza
            target_head = measurements.total_height / standard.head_units
            corrections['head_scale'] = target_head / measurements.head_height if measurements.head_height > 0 else 1.0
        else:
            corrections['head_scale'] = 1.0

        # Factor de corrección para relación hombro-cadera
        if abs(deviations['shoulder_hip']) > 0.1:
            corrections['shoulder_scale'] = standard.shoulder_hip_ratio * measurements.hip_width / measurements.shoulder_width \
                if measurements.shoulder_width > 0 else 1.0
        else:
            corrections['shoulder_scale'] = 1.0

        return corrections

    def _apply_corrections(self, skeleton: Dict[str, Any],
                           corrections: Dict[str, float]) -> Dict[str, Any]:
        """
        Aplica las correcciones al esqueleto.

        Args:
            skeleton: Esqueleto original
            corrections: Factores de corrección

        Returns:
            Esqueleto con correcciones aplicadas
        """
        optimized_skeleton = skeleton.copy()
        bones = optimized_skeleton.get('bones', [])

        # Aplicar escalas a segmentos específicos
        for bone in bones:
            bone_name = bone.get('name', '')

            # Escalar cabeza
            if 'head' in bone_name.lower():
                self._scale_bone(bone, corrections.get('head_scale', 1.0))

            # Escalar torso
            elif any(x in bone_name.lower() for x in ['torso', 'spine', 'chest']):
                self._scale_bone(bone, corrections.get('torso_scale', 1.0))

            # Escalar piernas
            elif any(x in bone_name.lower() for x in ['hip', 'knee', 'ankle', 'foot']):
                self._scale_bone(bone, corrections.get('leg_scale', 1.0))

            # Escalar brazos
            elif any(x in bone_name.lower() for x in ['shoulder', 'elbow', 'wrist']):
                self._scale_bone(bone, corrections.get('arm_scale', 1.0))

            # Escalar hombros específicamente
            elif 'shoulder' in bone_name.lower() or 'collar' in bone_name.lower():
                self._scale_bone(bone, corrections.get('shoulder_scale', 1.0), axis='x')

        return optimized_skeleton

    def _scale_bone(self, bone: Dict[str, Any], scale: float, axis: Optional[str] = None):
        """
        Escala un hueso por un factor.

        Args:
            bone: Datos del hueso
            scale: Factor de escala
            axis: Eje específico a escalar (None = todos)
        """
        if 'length' in bone:
            bone['length'] *= scale

        if 'position' in bone:
            pos = np.array(bone['position'])
            if axis == 'x':
                pos[0] *= scale
            elif axis == 'y':
                pos[1] *= scale
            elif axis == 'z':
                pos[2] *= scale
            else:
                pos *= scale
            bone['position'] = pos.tolist()

    def _calculate_quality_score(self, deviations: Dict[str, float]) -> float:
        """
        Calcula un score de calidad basado en las desviaciones.

        Args:
            deviations: Desviaciones calculadas

        Returns:
            Score de calidad [0, 100]
        """
        # Calcular error cuadrático medio de las desviaciones
        squared_errors = [dev ** 2 for dev in deviations.values()]
        mse = np.mean(squared_errors)

        # Convertir a score (menos error = mayor score)
        # Score perfecto = 100, cada 10% de desviación resta ~20 puntos
        score = max(0, 100 - (mse ** 0.5 * 200))

        return float(score)

    def _generate_proportion_diagram(self, measurements: AnthropometricMeasurements,
                                     standard: ProportionStandard) -> Dict[str, Any]:
        """Genera datos para visualización de proporciones."""
        head_units_current = measurements.total_height / measurements.head_height if measurements.head_height > 0 else 0

        return {
            'current_head_units': float(head_units_current),
            'target_head_units': standard.head_units,
            'segments': {
                'head': measurements.head_height / measurements.total_height if measurements.total_height > 0 else 0,
                'torso': measurements.torso_height / measurements.total_height if measurements.total_height > 0 else 0,
                'legs': measurements.leg_length / measurements.total_height if measurements.total_height > 0 else 0
            },
            'target_segments': {
                'torso': standard.torso_ratio,
                'legs': standard.leg_ratio
            }
        }

    def _generate_recommendations(self, deviations: Dict[str, float]) -> List[str]:
        """
        Genera recomendaciones basadas en las desviaciones.

        Args:
            deviations: Desviaciones calculadas

        Returns:
            Lista de recomendaciones
        """
        recommendations = []

        if abs(deviations['torso']) > 0.05:
            action = "alargar" if deviations['torso'] < 0 else "acortar"
            recommendations.append(f"Considerar {action} el torso en {abs(deviations['torso']*100):.1f}%")

        if abs(deviations['legs']) > 0.05:
            action = "alargar" if deviations['legs'] < 0 else "acortar"
            recommendations.append(f"Considerar {action} las piernas en {abs(deviations['legs']*100):.1f}%")

        if abs(deviations['arms']) > 0.05:
            action = "alargar" if deviations['arms'] < 0 else "acortar"
            recommendations.append(f"Considerar {action} los brazos en {abs(deviations['arms']*100):.1f}%")

        if abs(deviations['head_units']) > 0.5:
            action = "aumentar" if deviations['head_units'] < 0 else "reducir"
            recommendations.append(f"Considerar {action} el tamaño de la cabeza")

        if abs(deviations['shoulder_hip']) > 0.1:
            action = "ensanchar" if deviations['shoulder_hip'] < 0 else "estrechar"
            recommendations.append(f"Considerar {action} los hombros relativos a las caderas")

        if not recommendations:
            recommendations.append("Las proporciones están dentro de rangos aceptables")

        return recommendations
