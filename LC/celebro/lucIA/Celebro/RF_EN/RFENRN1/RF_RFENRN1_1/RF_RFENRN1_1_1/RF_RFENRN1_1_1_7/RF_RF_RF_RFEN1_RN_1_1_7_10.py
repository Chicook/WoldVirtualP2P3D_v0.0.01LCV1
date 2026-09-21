"""
Neurona 10: Métricas de Rendimiento y Analytics (Performance Metrics & Analytics)
Sistema avanzado de análisis de rendimiento para avatares 3D, optimización en tiempo real
y predicción de impacto en performance.

Librerías: numpy, pandas (opcional), time
Técnicas: Performance Profiling, GPU Metrics, Draw Call Optimization, Memory Analysis
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict


logger = logging.getLogger(__name__)


class PerformanceCategory(Enum):
    """Categorías de rendimiento para clasificación."""
    EXCELLENT = "excellent"  # >90 FPS
    GOOD = "good"           # 60-90 FPS
    ACCEPTABLE = "acceptable"  # 30-60 FPS
    POOR = "poor"           # 15-30 FPS
    UNPLAYABLE = "unplayable"  # <15 FPS


class OptimizationPriority(Enum):
    """Prioridades de optimización."""
    CRITICAL = 3
    HIGH = 2
    MEDIUM = 1
    LOW = 0


@dataclass
class GeometryMetrics:
    """Métricas de geometría de un avatar."""
    vertex_count: int
    triangle_count: int
    submesh_count: int
    bone_count: int
    blend_shape_count: int

    def estimate_draw_calls(self) -> int:
        """Estima el número de draw calls."""
        return max(1, self.submesh_count)

    def estimate_vertex_processing_cost(self) -> float:
        """Estima el costo de procesamiento de vértices."""
        base_cost = self.vertex_count * 0.001  # Costo base

        # Penalizar por huesos (skinning)
        skinning_cost = self.bone_count * 0.01

        # Penalizar por blend shapes
        blend_cost = self.blend_shape_count * 0.005

        return base_cost + skinning_cost + blend_cost


@dataclass
class TextureMetrics:
    """Métricas de texturas de un avatar."""
    texture_count: int
    total_texture_memory_mb: float
    max_texture_resolution: int
    has_mipmaps: bool
    compression_format: str

    def estimate_bandwidth_cost(self) -> float:
        """Estima el costo de ancho de banda de texturas."""
        bandwidth = self.total_texture_memory_mb

        if not self.has_mipmaps:
            bandwidth *= 1.33  # Penalizar sin mipmaps

        if self.compression_format == "uncompressed":
            bandwidth *= 2.0  # Penalizar texturas sin comprimir

        return bandwidth


@dataclass
class AnimationMetrics:
    """Métricas de animaciones de un avatar."""
    animation_count: int
    total_keyframes: int
    bone_tracks: int
    blend_shapes_animated: int

    def estimate_animation_cost(self) -> float:
        """Estima el costo de procesamiento de animaciones."""
        cost = 0.0

        # Costo de evaluación de keyframes
        cost += self.total_keyframes * 0.0001

        # Costo de aplicar transformaciones a huesos
        cost += self.bone_tracks * 0.001

        # Costo de blend shapes animados
        cost += self.blend_shapes_animated * 0.002

        return cost


@dataclass
class PhysicsMetrics:
    """Métricas de física de un avatar."""
    cloth_particle_count: int
    cloth_constraint_count: int
    collider_count: int

    def estimate_physics_cost(self) -> float:
        """Estima el costo de simulación física."""
        cost = 0.0

        # Costo de partículas de cloth
        cost += self.cloth_particle_count * 0.001

        # Costo de constraints
        cost += self.cloth_constraint_count * 0.0005

        # Costo de colisiones
        cost += self.collider_count * 0.01

        return cost


@dataclass
class PerformanceReport:
    """Reporte completo de rendimiento."""
    geometry_metrics: GeometryMetrics
    texture_metrics: TextureMetrics
    animation_metrics: AnimationMetrics
    physics_metrics: PhysicsMetrics

    overall_score: float = 0.0
    category: PerformanceCategory = PerformanceCategory.ACCEPTABLE
    estimated_fps: float = 60.0
    estimated_memory_mb: float = 0.0

    optimization_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)


class PerformanceMetricsNeuron:
    """
    Neurona especializada en análisis de rendimiento y optimización de avatares 3D.

    Funcionalidades:
    - Análisis exhaustivo de métricas de rendimiento
    - Estimación de FPS y uso de memoria
    - Detección de cuellos de botella
    - Generación de sugerencias de optimización
    - Comparación con benchmarks
    - Predicción de rendimiento en diferentes hardware
    - Análisis de draw calls y batching
    - Optimización automática de assets
    """

    def __init__(self):
        """Inicializa la neurona de métricas de rendimiento."""
        self.name = "PerformanceMetricsNeuron"
        self.version = "1.0.0"

        # Benchmarks de referencia
        self.hardware_profiles = self._load_hardware_profiles()

        # Umbrales de rendimiento
        self.thresholds = {
            'max_vertices': 75000,
            'max_triangles': 50000,
            'max_bones': 150,
            'max_texture_memory_mb': 512,
            'max_draw_calls': 10,
            'max_blend_shapes': 50
        }

        logger.info(f"{self.name} v{self.version} inicializada")

    def _load_hardware_profiles(self) -> Dict[str, Dict[str, float]]:
        """Carga perfiles de hardware de referencia."""
        return {
            'high_end': {
                'vertex_throughput': 1000000,  # Vértices por frame
                'texture_bandwidth_gb': 10.0,
                'shader_complexity': 1.0,
                'base_fps': 144.0
            },
            'mid_range': {
                'vertex_throughput': 500000,
                'texture_bandwidth_gb': 5.0,
                'shader_complexity': 0.7,
                'base_fps': 60.0
            },
            'low_end': {
                'vertex_throughput': 200000,
                'texture_bandwidth_gb': 2.0,
                'shader_complexity': 0.5,
                'base_fps': 30.0
            },
            'mobile': {
                'vertex_throughput': 100000,
                'texture_bandwidth_gb': 1.0,
                'shader_complexity': 0.3,
                'base_fps': 30.0
            }
        }

    def process(self, avatar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa y analiza el rendimiento de un avatar.

        Args:
            avatar_data: Diccionario con todos los datos del avatar

        Returns:
            Diccionario con reporte de rendimiento completo
        """
        try:
            start_time = time.time()

            # Extraer métricas de cada componente
            geometry_metrics = self._extract_geometry_metrics(avatar_data)
            texture_metrics = self._extract_texture_metrics(avatar_data)
            animation_metrics = self._extract_animation_metrics(avatar_data)
            physics_metrics = self._extract_physics_metrics(avatar_data)

            # Calcular costos individuales
            geometry_cost = geometry_metrics.estimate_vertex_processing_cost()
            texture_cost = texture_metrics.estimate_bandwidth_cost()
            animation_cost = animation_metrics.estimate_animation_cost()
            physics_cost = physics_metrics.estimate_physics_cost()

            # Calcular costo total
            total_cost = geometry_cost + texture_cost + animation_cost + physics_cost

            # Estimar FPS para diferentes perfiles de hardware
            fps_estimates = self._estimate_fps_by_hardware(
                geometry_metrics, texture_metrics, animation_metrics, physics_metrics
            )

            # Estimar uso de memoria
            memory_estimate = self._estimate_memory_usage(
                geometry_metrics, texture_metrics, animation_metrics, physics_metrics
            )

            # Detectar cuellos de botella
            bottlenecks = self._detect_bottlenecks(
                geometry_metrics, texture_metrics, animation_metrics, physics_metrics
            )

            # Generar sugerencias de optimización
            suggestions = self._generate_optimization_suggestions(
                geometry_metrics, texture_metrics, animation_metrics,
                physics_metrics, bottlenecks
            )

            # Calcular score general
            overall_score = self._calculate_overall_score(
                geometry_metrics, texture_metrics, animation_metrics, physics_metrics
            )

            # Categorizar rendimiento
            category = self._categorize_performance(fps_estimates['mid_range'])

            # Crear reporte
            report = PerformanceReport(
                geometry_metrics=geometry_metrics,
                texture_metrics=texture_metrics,
                animation_metrics=animation_metrics,
                physics_metrics=physics_metrics,
                overall_score=overall_score,
                category=category,
                estimated_fps=fps_estimates['mid_range'],
                estimated_memory_mb=memory_estimate,
                optimization_suggestions=suggestions,
                bottlenecks=bottlenecks
            )

            # Análisis de profiling
            profiling_time = time.time() - start_time

            return {
                'success': True,
                'report': self._report_to_dict(report),
                'fps_estimates': fps_estimates,
                'cost_breakdown': {
                    'geometry': float(geometry_cost),
                    'textures': float(texture_cost),
                    'animations': float(animation_cost),
                    'physics': float(physics_cost),
                    'total': float(total_cost)
                },
                'compliance': self._check_threshold_compliance(report),
                'profiling_time_ms': profiling_time * 1000,
                'optimization_impact': self._estimate_optimization_impact(suggestions)
            }

        except Exception as e:
            logger.error(f"Error en {self.name}: {e}")
            return {'error': str(e)}

    def _extract_geometry_metrics(self, avatar_data: Dict[str, Any]) -> GeometryMetrics:
        """Extrae métricas de geometría del avatar."""
        mesh_data = avatar_data.get('mesh', {})
        skeleton = avatar_data.get('skeleton', {})

        vertices = mesh_data.get('vertices', [])
        faces = mesh_data.get('faces', [])
        bones = skeleton.get('bones', [])

        # Contar submeshes (estimación)
        submesh_count = len(mesh_data.get('materials', [])) or 1

        # Contar blend shapes
        blend_shapes = avatar_data.get('blend_shapes', [])

        return GeometryMetrics(
            vertex_count=len(vertices),
            triangle_count=len(faces),
            submesh_count=submesh_count,
            bone_count=len(bones),
            blend_shape_count=len(blend_shapes)
        )

    def _extract_texture_metrics(self, avatar_data: Dict[str, Any]) -> TextureMetrics:
        """Extrae métricas de texturas del avatar."""
        textures = avatar_data.get('textures', {})

        texture_count = len(textures)
        total_memory = 0.0
        max_resolution = 0

        for texture_name, texture_data in textures.items():
            # Estimar tamaño de textura
            if isinstance(texture_data, dict):
                width = texture_data.get('width', 1024)
                height = texture_data.get('height', 1024)
                channels = texture_data.get('channels', 4)

                # Memoria sin comprimir en MB
                memory = (width * height * channels) / (1024 * 1024)

                # Si hay compresión, reducir estimación
                if texture_data.get('compressed', False):
                    memory /= 4  # DXT compression ~4:1

                total_memory += memory
                max_resolution = max(max_resolution, max(width, height))

        return TextureMetrics(
            texture_count=texture_count,
            total_texture_memory_mb=float(total_memory),
            max_texture_resolution=max_resolution,
            has_mipmaps=True,  # Asumimos que sí por defecto
            compression_format="compressed" if total_memory < 100 else "uncompressed"
        )

    def _extract_animation_metrics(self, avatar_data: Dict[str, Any]) -> AnimationMetrics:
        """Extrae métricas de animaciones del avatar."""
        animations = avatar_data.get('animations', [])

        animation_count = len(animations)
        total_keyframes = 0
        bone_tracks = 0
        blend_shapes_animated = 0

        for anim in animations:
            tracks = anim.get('tracks', [])
            for track in tracks:
                keyframes = track.get('keyframes', [])
                total_keyframes += len(keyframes)
                bone_tracks += 1

        # Estimar blend shapes animados
        blend_shapes_animated = len(avatar_data.get('blend_shapes', [])) // 2

        return AnimationMetrics(
            animation_count=animation_count,
            total_keyframes=total_keyframes,
            bone_tracks=bone_tracks,
            blend_shapes_animated=blend_shapes_animated
        )

    def _extract_physics_metrics(self, avatar_data: Dict[str, Any]) -> PhysicsMetrics:
        """Extrae métricas de física del avatar."""
        cloth_data = avatar_data.get('cloth', [])

        particle_count = 0
        constraint_count = 0

        for cloth_item in cloth_data:
            particle_count += cloth_item.get('particle_count', 0)
            constraint_count += cloth_item.get('spring_count', 0)

        collider_count = len(avatar_data.get('colliders', []))

        return PhysicsMetrics(
            cloth_particle_count=particle_count,
            cloth_constraint_count=constraint_count,
            collider_count=collider_count
        )

    def _estimate_fps_by_hardware(self, geometry: GeometryMetrics,
                                  textures: TextureMetrics,
                                  animations: AnimationMetrics,
                                  physics: PhysicsMetrics) -> Dict[str, float]:
        """Estima FPS para diferentes perfiles de hardware."""
        estimates = {}

        for profile_name, profile in self.hardware_profiles.items():
            # Calcular factor de geometría
            geometry_factor = min(1.0, profile['vertex_throughput'] /
                                  max(1, geometry.vertex_count))

            # Calcular factor de texturas
            texture_factor = min(1.0, profile['texture_bandwidth_gb'] * 1024 /
                                 max(1, textures.total_texture_memory_mb))

            # Calcular factor de física
            physics_factor = 1.0 - (physics.cloth_particle_count / 10000) * 0.5
            physics_factor = max(0.1, physics_factor)

            # FPS estimado
            base_fps = profile['base_fps']
            estimated_fps = base_fps * geometry_factor * texture_factor * physics_factor

            estimates[profile_name] = float(estimated_fps)

        return estimates

    def _estimate_memory_usage(self, geometry: GeometryMetrics,
                               textures: TextureMetrics,
                               animations: AnimationMetrics,
                               physics: PhysicsMetrics) -> float:
        """Estima el uso total de memoria."""
        memory_mb = 0.0

        # Geometría: 12 bytes por vértice (posición) + 12 bytes (normal) + 8 bytes (UV)
        geometry_mb = (geometry.vertex_count * 32) / (1024 * 1024)
        memory_mb += geometry_mb

        # Índices: 2 bytes por índice (asumiendo uint16)
        indices_mb = (geometry.triangle_count * 3 * 2) / (1024 * 1024)
        memory_mb += indices_mb

        # Texturas
        memory_mb += textures.total_texture_memory_mb

        # Skeleton y animaciones (estimación)
        skeleton_mb = geometry.bone_count * 64 / (1024 * 1024)  # 64 bytes por hueso
        memory_mb += skeleton_mb

        animation_mb = animations.total_keyframes * 40 / (1024 * 1024)  # 40 bytes por keyframe
        memory_mb += animation_mb

        # Física
        physics_mb = physics.cloth_particle_count * 48 / (1024 * 1024)  # 48 bytes por partícula
        memory_mb += physics_mb

        return float(memory_mb)

    def _detect_bottlenecks(self, geometry: GeometryMetrics,
                            textures: TextureMetrics,
                            animations: AnimationMetrics,
                            physics: PhysicsMetrics) -> List[str]:
        """Detecta cuellos de botella en el rendimiento."""
        bottlenecks = []

        if geometry.vertex_count > self.thresholds['max_vertices']:
            bottlenecks.append(f"Alto número de vértices ({geometry.vertex_count} > {self.thresholds['max_vertices']})")

        if geometry.triangle_count > self.thresholds['max_triangles']:
            bottlenecks.append(f"Alto número de triángulos ({geometry.triangle_count} > {self.thresholds['max_triangles']})")

        if geometry.bone_count > self.thresholds['max_bones']:
            bottlenecks.append(f"Alto número de huesos ({geometry.bone_count} > {self.thresholds['max_bones']})")

        if textures.total_texture_memory_mb > self.thresholds['max_texture_memory_mb']:
            bottlenecks.append(f"Alto uso de memoria de texturas ({textures.total_texture_memory_mb:.1f}MB > {self.thresholds['max_texture_memory_mb']}MB)")

        draw_calls = geometry.estimate_draw_calls()
        if draw_calls > self.thresholds['max_draw_calls']:
            bottlenecks.append(f"Alto número de draw calls ({draw_calls} > {self.thresholds['max_draw_calls']})")

        if physics.cloth_particle_count > 1000:
            bottlenecks.append(f"Alto número de partículas de cloth ({physics.cloth_particle_count} > 1000)")

        return bottlenecks

    def _generate_optimization_suggestions(self, geometry: GeometryMetrics,
                                           textures: TextureMetrics,
                                           animations: AnimationMetrics,
                                           physics: PhysicsMetrics,
                                           bottlenecks: List[str]) -> List[Dict[str, Any]]:
        """Genera sugerencias de optimización priorizadas."""
        suggestions = []

        # Optimizaciones de geometría
        if geometry.vertex_count > self.thresholds['max_vertices']:
            suggestions.append({
                'category': 'geometry',
                'priority': OptimizationPriority.HIGH.value,
                'title': 'Reducir número de vértices',
                'description': f'Aplicar LOD o simplificación de malla',
                'potential_gain': '30-50% mejora en rendimiento',
                'estimated_reduction': geometry.vertex_count - self.thresholds['max_vertices']
            })

        if geometry.triangle_count > self.thresholds['max_triangles']:
            suggestions.append({
                'category': 'geometry',
                'priority': OptimizationPriority.HIGH.value,
                'title': 'Simplificar geometría',
                'description': 'Usar herramientas de decimación de mallas',
                'potential_gain': '25-40% mejora en rendimiento'
            })

        # Optimizaciones de texturas
        if textures.max_texture_resolution > 2048:
            suggestions.append({
                'category': 'textures',
                'priority': OptimizationPriority.MEDIUM.value,
                'title': 'Reducir resolución de texturas',
                'description': 'Usar texturas de máximo 2048x2048',
                'potential_gain': '20-30% reducción de memoria'
            })

        if textures.compression_format == "uncompressed":
            suggestions.append({
                'category': 'textures',
                'priority': OptimizationPriority.HIGH.value,
                'title': 'Comprimir texturas',
                'description': 'Usar formato DXT/BC para texturas',
                'potential_gain': '75% reducción de memoria'
            })

        if not textures.has_mipmaps:
            suggestions.append({
                'category': 'textures',
                'priority': OptimizationPriority.MEDIUM.value,
                'title': 'Generar mipmaps',
                'description': 'Añadir mipmaps a todas las texturas',
                'potential_gain': '15-25% mejora en performance'
            })

        # Optimizaciones de animaciones
        if animations.total_keyframes > 1000:
            suggestions.append({
                'category': 'animations',
                'priority': OptimizationPriority.LOW.value,
                'title': 'Reducir keyframes de animaciones',
                'description': 'Aplicar compresión de animaciones',
                'potential_gain': '10-15% mejora en CPU'
            })

        # Optimizaciones de física
        if physics.cloth_particle_count > 500:
            suggestions.append({
                'category': 'physics',
                'priority': OptimizationPriority.MEDIUM.value,
                'title': 'Reducir resolución de cloth',
                'description': 'Usar menos partículas en simulación de tela',
                'potential_gain': '40-60% mejora en física'
            })

        return suggestions

    def _calculate_overall_score(self, geometry: GeometryMetrics,
                                 textures: TextureMetrics,
                                 animations: AnimationMetrics,
                                 physics: PhysicsMetrics) -> float:
        """Calcula un score general de rendimiento [0-100]."""
        score = 100.0

        # Penalizar por vértices excesivos
        if geometry.vertex_count > self.thresholds['max_vertices']:
            penalty = (geometry.vertex_count - self.thresholds['max_vertices']) / self.thresholds['max_vertices'] * 30
            score -= min(penalty, 30)

        # Penalizar por texturas pesadas
        if textures.total_texture_memory_mb > self.thresholds['max_texture_memory_mb']:
            penalty = (textures.total_texture_memory_mb - self.thresholds['max_texture_memory_mb']) / \
                self.thresholds['max_texture_memory_mb'] * 25
            score -= min(penalty, 25)

        # Penalizar por física compleja
        if physics.cloth_particle_count > 500:
            penalty = (physics.cloth_particle_count - 500) / 500 * 15
            score -= min(penalty, 15)

        # Penalizar por muchos huesos
        if geometry.bone_count > self.thresholds['max_bones']:
            penalty = (geometry.bone_count - self.thresholds['max_bones']) / self.thresholds['max_bones'] * 10
            score -= min(penalty, 10)

        return max(0.0, float(score))

    def _categorize_performance(self, estimated_fps: float) -> PerformanceCategory:
        """Categoriza el rendimiento basado en FPS estimado."""
        if estimated_fps >= 90:
            return PerformanceCategory.EXCELLENT
        elif estimated_fps >= 60:
            return PerformanceCategory.GOOD
        elif estimated_fps >= 30:
            return PerformanceCategory.ACCEPTABLE
        elif estimated_fps >= 15:
            return PerformanceCategory.POOR
        else:
            return PerformanceCategory.UNPLAYABLE

    def _check_threshold_compliance(self, report: PerformanceReport) -> Dict[str, bool]:
        """Verifica cumplimiento de umbrales de rendimiento."""
        return {
            'vertices': report.geometry_metrics.vertex_count <= self.thresholds['max_vertices'],
            'triangles': report.geometry_metrics.triangle_count <= self.thresholds['max_triangles'],
            'bones': report.geometry_metrics.bone_count <= self.thresholds['max_bones'],
            'texture_memory': report.texture_metrics.total_texture_memory_mb <= self.thresholds['max_texture_memory_mb'],
            'draw_calls': report.geometry_metrics.estimate_draw_calls() <= self.thresholds['max_draw_calls']
        }

    def _estimate_optimization_impact(self, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estima el impacto de aplicar todas las optimizaciones sugeridas."""
        total_fps_gain = 0.0
        total_memory_reduction = 0.0

        for suggestion in suggestions:
            potential_gain = suggestion.get('potential_gain', '')

            # Extraer porcentajes de las descripciones
            if 'fps' in potential_gain.lower() or 'rendimiento' in potential_gain.lower():
                # Estimación aproximada
                total_fps_gain += 20.0

            if 'memoria' in potential_gain.lower() or 'memory' in potential_gain.lower():
                total_memory_reduction += 15.0

        return {
            'estimated_fps_improvement': f"+{min(total_fps_gain, 100):.0f}%",
            'estimated_memory_reduction': f"-{min(total_memory_reduction, 80):.0f}%",
            'priority_fixes': len([s for s in suggestions if s['priority'] >= OptimizationPriority.HIGH.value])
        }

    def _report_to_dict(self, report: PerformanceReport) -> Dict[str, Any]:
        """Convierte un PerformanceReport a diccionario."""
        return {
            'geometry': {
                'vertex_count': report.geometry_metrics.vertex_count,
                'triangle_count': report.geometry_metrics.triangle_count,
                'bone_count': report.geometry_metrics.bone_count,
                'blend_shape_count': report.geometry_metrics.blend_shape_count,
                'draw_calls': report.geometry_metrics.estimate_draw_calls()
            },
            'textures': {
                'count': report.texture_metrics.texture_count,
                'memory_mb': report.texture_metrics.total_texture_memory_mb,
                'max_resolution': report.texture_metrics.max_texture_resolution,
                'has_mipmaps': report.texture_metrics.has_mipmaps,
                'compression': report.texture_metrics.compression_format
            },
            'animations': {
                'count': report.animation_metrics.animation_count,
                'total_keyframes': report.animation_metrics.total_keyframes,
                'bone_tracks': report.animation_metrics.bone_tracks
            },
            'physics': {
                'cloth_particles': report.physics_metrics.cloth_particle_count,
                'constraints': report.physics_metrics.cloth_constraint_count,
                'colliders': report.physics_metrics.collider_count
            },
            'overall_score': report.overall_score,
            'category': report.category.value,
            'estimated_fps': report.estimated_fps,
            'estimated_memory_mb': report.estimated_memory_mb,
            'bottlenecks': report.bottlenecks,
            'optimization_suggestions': report.optimization_suggestions
        }
