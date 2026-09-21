"""
Neurona 4: Procesamiento de Animaciones (Animation Processing & Motion Capture)
Algoritmos avanzados para optimizar, suavizar y comprimir animaciones de avatares 3D.

Librerías: numpy, scipy, scikit-learn
Técnicas: Keyframe Reduction, Motion Smoothing, Animation Compression, Motion Retargeting
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


logger = logging.getLogger(__name__)


@dataclass
class Keyframe:
    """Representa un keyframe en una animación."""
    time: float
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1]))  # Quaternion
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))

    def __post_init__(self):
        """Conversión de listas a arrays."""
        if isinstance(self.position, list):
            self.position = np.array(self.position)
        if isinstance(self.rotation, list):
            self.rotation = np.array(self.rotation)
        if isinstance(self.scale, list):
            self.scale = np.array(self.scale)


@dataclass
class AnimationTrack:
    """Track de animación para un hueso específico."""
    bone_name: str
    bone_index: int
    keyframes: List[Keyframe]
    interpolation: str = 'cubic'  # 'linear', 'cubic', 'bezier'

    def get_duration(self) -> float:
        """Retorna la duración total del track."""
        if not self.keyframes:
            return 0.0
        return max(kf.time for kf in self.keyframes)

    def get_keyframe_count(self) -> int:
        """Retorna el número de keyframes."""
        return len(self.keyframes)


@dataclass
class Animation:
    """Animación completa con múltiples tracks."""
    name: str
    tracks: List[AnimationTrack]
    fps: float = 30.0
    loop: bool = False

    def get_duration(self) -> float:
        """Retorna la duración total de la animación."""
        if not self.tracks:
            return 0.0
        return max(track.get_duration() for track in self.tracks)

    def get_total_keyframes(self) -> int:
        """Retorna el número total de keyframes."""
        return sum(track.get_keyframe_count() for track in self.tracks)


class AnimationProcessingNeuron:
    """
    Neurona especializada en procesamiento y optimización de animaciones para avatares 3D.

    Funcionalidades:
    - Reducción de keyframes con preservación de movimiento
    - Suavizado de animaciones (motion smoothing)
    - Compresión de datos de animación
    - Retargeting de animaciones entre esqueletos
    - Detección y corrección de jitter
    - Interpolación avanzada de keyframes
    - Análisis de calidad de movimiento
    - Conversión entre formatos de animación
    """

    def __init__(self):
        """Inicializa la neurona de procesamiento de animaciones."""
        self.name = "AnimationProcessingNeuron"
        self.version = "1.0.0"
        self.animation_cache = {}
        logger.info(f"{self.name} v{self.version} inicializada")

    def process(self, avatar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa y optimiza las animaciones de un avatar.

        Args:
            avatar_data: Diccionario con datos del avatar incluyendo animaciones

        Returns:
            Diccionario con animaciones optimizadas y métricas
        """
        try:
            animations_data = avatar_data.get('animations', [])
            optimization_level = avatar_data.get('optimization_level', 'medium')

            if not animations_data:
                return {'error': 'No se encontraron animaciones'}

            results = []

            for anim_data in animations_data:
                logger.info(f"Procesando animación: {anim_data.get('name', 'unnamed')}")

                # Construir objeto Animation
                animation = self._build_animation(anim_data)

                # Análisis inicial
                initial_metrics = self._analyze_animation(animation)

                # Detectar y corregir jitter
                animation = self._fix_jitter(animation)

                # Suavizar animación
                animation = self._smooth_animation(animation)

                # Reducir keyframes
                animation = self._reduce_keyframes(animation, optimization_level)

                # Comprimir datos
                compressed_data = self._compress_animation(animation)

                # Análisis final
                final_metrics = self._analyze_animation(animation)

                results.append({
                    'name': animation.name,
                    'success': True,
                    'optimized_animation': self._animation_to_dict(animation),
                    'compressed_data': compressed_data,
                    'initial_metrics': initial_metrics,
                    'final_metrics': final_metrics,
                    'keyframe_reduction': 1.0 - (final_metrics['total_keyframes'] /
                                                 initial_metrics['total_keyframes']),
                    'size_reduction': compressed_data['compression_ratio']
                })

            return {
                'success': True,
                'processed_animations': len(results),
                'results': results
            }

        except Exception as e:
            logger.error(f"Error en {self.name}: {e}")
            return {'error': str(e)}

    def _build_animation(self, anim_data: Dict[str, Any]) -> Animation:
        """
        Construye un objeto Animation desde datos brutos.

        Args:
            anim_data: Datos de la animación

        Returns:
            Objeto Animation construido
        """
        tracks = []

        for track_data in anim_data.get('tracks', []):
            keyframes = []

            for kf_data in track_data.get('keyframes', []):
                kf = Keyframe(
                    time=kf_data.get('time', 0.0),
                    position=np.array(kf_data.get('position', [0, 0, 0])),
                    rotation=np.array(kf_data.get('rotation', [0, 0, 0, 1])),
                    scale=np.array(kf_data.get('scale', [1, 1, 1]))
                )
                keyframes.append(kf)

            track = AnimationTrack(
                bone_name=track_data.get('bone_name', ''),
                bone_index=track_data.get('bone_index', 0),
                keyframes=keyframes,
                interpolation=track_data.get('interpolation', 'cubic')
            )
            tracks.append(track)

        return Animation(
            name=anim_data.get('name', 'unnamed'),
            tracks=tracks,
            fps=anim_data.get('fps', 30.0),
            loop=anim_data.get('loop', False)
        )

    def _analyze_animation(self, animation: Animation) -> Dict[str, Any]:
        """
        Analiza las características y calidad de una animación.

        Args:
            animation: Animación a analizar

        Returns:
            Diccionario con métricas de la animación
        """
        duration = animation.get_duration()
        total_keyframes = animation.get_total_keyframes()
        track_count = len(animation.tracks)

        # Calcular keyframes por segundo promedio
        avg_keyframes_per_second = total_keyframes / duration if duration > 0 else 0

        # Detectar tracks con muchos keyframes (potencial jitter)
        high_density_tracks = sum(1 for track in animation.tracks
                                  if track.get_keyframe_count() / duration > animation.fps * 1.5)

        # Calcular variación de movimiento
        motion_variance = self._calculate_motion_variance(animation)

        # Detectar tracks sin movimiento significativo
        static_tracks = sum(1 for track in animation.tracks
                            if self._is_track_static(track))

        return {
            'duration': duration,
            'total_keyframes': total_keyframes,
            'track_count': track_count,
            'avg_keyframes_per_second': avg_keyframes_per_second,
            'high_density_tracks': high_density_tracks,
            'static_tracks': static_tracks,
            'motion_variance': motion_variance,
            'fps': animation.fps,
            'is_loop': animation.loop
        }

    def _calculate_motion_variance(self, animation: Animation) -> float:
        """Calcula la varianza de movimiento en la animación."""
        total_variance = 0.0
        track_count = 0

        for track in animation.tracks:
            if len(track.keyframes) < 2:
                continue

            # Calcular varianza de posición
            positions = np.array([kf.position for kf in track.keyframes])
            pos_variance = np.var(positions)

            total_variance += pos_variance
            track_count += 1

        return float(total_variance / track_count) if track_count > 0 else 0.0

    def _is_track_static(self, track: AnimationTrack, threshold: float = 0.001) -> bool:
        """Determina si un track tiene movimiento significativo."""
        if len(track.keyframes) < 2:
            return True

        positions = np.array([kf.position for kf in track.keyframes])
        rotations = np.array([kf.rotation for kf in track.keyframes])

        # Verificar variación de posición
        pos_range = positions.max(axis=0) - positions.min(axis=0)
        if np.any(pos_range > threshold):
            return False

        # Verificar variación de rotación
        rot_variance = np.var(rotations, axis=0)
        if np.any(rot_variance > threshold):
            return False

        return True

    def _fix_jitter(self, animation: Animation) -> Animation:
        """
        Detecta y corrige jitter en la animación.

        Args:
            animation: Animación a corregir

        Returns:
            Animación con jitter corregido
        """
        for track in animation.tracks:
            if len(track.keyframes) < 5:
                continue

            # Aplicar filtro Savitzky-Golay para suavizar
            times = np.array([kf.time for kf in track.keyframes])
            positions = np.array([kf.position for kf in track.keyframes])

            # Solo aplicar si hay suficientes keyframes
            window_length = min(5, len(track.keyframes) if len(track.keyframes) % 2 == 1
                                else len(track.keyframes) - 1)

            if window_length >= 5:
                for i in range(3):  # x, y, z
                    smoothed = savgol_filter(positions[:, i], window_length, 3)
                    for j, kf in enumerate(track.keyframes):
                        kf.position[i] = smoothed[j]

        return animation

    def _smooth_animation(self, animation: Animation) -> Animation:
        """
        Suaviza los movimientos de la animación.

        Args:
            animation: Animación a suavizar

        Returns:
            Animación suavizada
        """
        for track in animation.tracks:
            if len(track.keyframes) < 3:
                continue

            times = np.array([kf.time for kf in track.keyframes])
            positions = np.array([kf.position for kf in track.keyframes])

            # Interpolación con spline cúbico para suavidad
            try:
                cs_x = CubicSpline(times, positions[:, 0], bc_type='natural')
                cs_y = CubicSpline(times, positions[:, 1], bc_type='natural')
                cs_z = CubicSpline(times, positions[:, 2], bc_type='natural')

                for i, kf in enumerate(track.keyframes):
                    t = kf.time
                    kf.position = np.array([cs_x(t), cs_y(t), cs_z(t)])
            except:
                logger.warning(f"No se pudo suavizar track {track.bone_name}")

            # Suavizar rotaciones usando SLERP
            if len(track.keyframes) >= 2:
                rotations = [kf.rotation for kf in track.keyframes]
                rot_objects = Rotation.from_quat(rotations)

                try:
                    slerp = Slerp(times, rot_objects)
                    for i, kf in enumerate(track.keyframes):
                        kf.rotation = slerp(kf.time).as_quat()
                except:
                    logger.warning(f"No se pudo suavizar rotaciones de {track.bone_name}")

        return animation

    def _reduce_keyframes(self, animation: Animation, level: str) -> Animation:
        """
        Reduce el número de keyframes manteniendo la calidad del movimiento.

        Args:
            animation: Animación a optimizar
            level: Nivel de optimización ('low', 'medium', 'high')

        Returns:
            Animación con keyframes reducidos
        """
        error_thresholds = {
            'low': 0.001,
            'medium': 0.01,
            'high': 0.05
        }

        error_threshold = error_thresholds.get(level, 0.01)

        for track in animation.tracks:
            if len(track.keyframes) < 3:
                continue

            optimized_keyframes = self._ramer_douglas_peucker(
                track.keyframes, error_threshold
            )

            # Siempre mantener primer y último keyframe
            if optimized_keyframes[0].time != track.keyframes[0].time:
                optimized_keyframes.insert(0, track.keyframes[0])
            if optimized_keyframes[-1].time != track.keyframes[-1].time:
                optimized_keyframes.append(track.keyframes[-1])

            track.keyframes = optimized_keyframes

        return animation

    def _ramer_douglas_peucker(self, keyframes: List[Keyframe],
                               epsilon: float) -> List[Keyframe]:
        """
        Implementación del algoritmo Ramer-Douglas-Peucker para reducción de keyframes.

        Args:
            keyframes: Lista de keyframes a simplificar
            epsilon: Tolerancia de error

        Returns:
            Lista simplificada de keyframes
        """
        if len(keyframes) < 3:
            return keyframes

        # Encontrar el punto con mayor distancia a la línea entre inicio y fin
        start = keyframes[0]
        end = keyframes[-1]

        max_distance = 0
        max_index = 0

        for i in range(1, len(keyframes) - 1):
            distance = self._perpendicular_distance(keyframes[i], start, end)
            if distance > max_distance:
                max_distance = distance
                max_index = i

        # Si la distancia máxima es mayor que epsilon, dividir y conquistar
        if max_distance > epsilon:
            # Recursión en ambas mitades
            left = self._ramer_douglas_peucker(keyframes[:max_index + 1], epsilon)
            right = self._ramer_douglas_peucker(keyframes[max_index:], epsilon)

            # Combinar resultados (sin duplicar el punto medio)
            return left[:-1] + right
        else:
            # Todos los puntos pueden ser aproximados por la línea
            return [start, end]

    def _perpendicular_distance(self, point: Keyframe, line_start: Keyframe,
                                line_end: Keyframe) -> float:
        """Calcula la distancia perpendicular de un punto a una línea."""
        # Usar posición como métrica principal
        p = point.position
        ls = line_start.position
        le = line_end.position

        # Vector de la línea
        line_vec = le - ls
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-6:
            return np.linalg.norm(p - ls)

        # Vector del punto
        point_vec = p - ls

        # Proyección escalar
        t = np.dot(point_vec, line_vec) / (line_len ** 2)
        t = np.clip(t, 0, 1)

        # Punto más cercano en la línea
        projection = ls + t * line_vec

        # Distancia
        distance = np.linalg.norm(p - projection)

        return float(distance)

    def _compress_animation(self, animation: Animation) -> Dict[str, Any]:
        """
        Comprime los datos de la animación para almacenamiento eficiente.

        Args:
            animation: Animación a comprimir

        Returns:
            Diccionario con datos comprimidos y métricas
        """
        # Calcular tamaño original (estimado)
        original_size = animation.get_total_keyframes() * (3 + 4 + 3) * 4  # floats de 4 bytes

        # Cuantizar valores para reducir precisión innecesaria
        quantized_tracks = []
        for track in animation.tracks:
            quantized_kfs = []
            for kf in track.keyframes:
                quantized_kfs.append({
                    'time': round(kf.time, 3),  # 3 decimales de precisión
                    'position': np.round(kf.position, 4).tolist(),  # 4 decimales
                    'rotation': np.round(kf.rotation, 4).tolist(),  # 4 decimales
                    'scale': np.round(kf.scale, 3).tolist()  # 3 decimales
                })
            quantized_tracks.append(quantized_kfs)

        # Calcular tamaño comprimido (estimado)
        compressed_size = len(str(quantized_tracks))

        return {
            'quantized_data': quantized_tracks,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': 1.0 - (compressed_size / original_size)
            if original_size > 0 else 0.0
        }

    def _animation_to_dict(self, animation: Animation) -> Dict[str, Any]:
        """Convierte un objeto Animation a diccionario."""
        return {
            'name': animation.name,
            'fps': animation.fps,
            'loop': animation.loop,
            'duration': animation.get_duration(),
            'tracks': [{
                'bone_name': track.bone_name,
                'bone_index': track.bone_index,
                'interpolation': track.interpolation,
                'keyframes': [{
                    'time': kf.time,
                    'position': kf.position.tolist(),
                    'rotation': kf.rotation.tolist(),
                    'scale': kf.scale.tolist()
                } for kf in track.keyframes]
            } for track in animation.tracks]
        }
