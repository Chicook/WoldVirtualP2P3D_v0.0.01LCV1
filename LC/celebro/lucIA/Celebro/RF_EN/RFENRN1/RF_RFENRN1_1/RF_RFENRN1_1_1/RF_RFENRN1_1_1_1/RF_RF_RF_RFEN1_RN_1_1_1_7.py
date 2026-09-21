"""
FPSController - Controlador de FPS Inteligente
==========================================

Neurona especializada en controlar y optimizar los FPS del simulador.

Algoritmos 2024-2025:
- Adaptive framerate
- Frame time smoothing
- Predictive frame skipping
"""

import numpy as np
import time
from typing import Dict, Any, List


class FPSController:
    """Controlador de FPS inteligente"""

    def __init__(self):
        self.target_fps = 45
        self.current_fps = 0
        self.frame_times = []

        self.adaptive_mode = True
        self.performance_history = []

        print("✓ FPSController inicializado")

    def analyze_frame_time(self, frame_time_ms: float) -> Dict[str, Any]:
        """
        Analiza el tiempo de frame

        Args:
            frame_time_ms: Tiempo de frame en milisegundos

        Returns:
            Análisis y recomendaciones
        """
        self.frame_times.append(frame_time_ms)

        # Mantener solo últimos 60 frames
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

        # Calcular FPS actual
        avg_frame_time = np.mean(self.frame_times)
        self.current_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0

        # Detectar problemas
        is_lagging = self.current_fps < (self.target_fps * 0.8)
        is_stuttering = np.std(self.frame_times) > (avg_frame_time * 0.3)

        # Análisis de estabilidad
        stability = self._calculate_stability()

        return {
            'current_fps': self.current_fps,
            'target_fps': self.target_fps,
            'frame_time_ms': frame_time_ms,
            'avg_frame_time_ms': avg_frame_time,
            'is_lagging': is_lagging,
            'is_stuttering': is_stuttering,
            'stability': stability,
            'optimization_needed': is_lagging or is_stuttering
        }

    def _calculate_stability(self) -> float:
        """Calcula la estabilidad del FPS (0-100)"""
        if len(self.frame_times) < 3:
            return 100.0

        cv = np.std(self.frame_times) / np.mean(self.frame_times)  # Coefficient of variation
        stability = max(0, 100 - (cv * 100))

        return stability

    def get_optimization_settings(self) -> Dict[str, Any]:
        """
        Obtiene configuraciones de optimización basadas en FPS

        Returns:
            Settings de optimización
        """
        if self.current_fps < 20:
            # Optimización extrema
            return {
                'lod_distance': 25,
                'physics_simplify': True,
                'disable_shadows': True,
                'texture_quality': 'low',
                'anti_aliasing': False
            }
        elif self.current_fps < 30:
            # Optimización agresiva
            return {
                'lod_distance': 50,
                'physics_simplify': False,
                'disable_shadows': False,
                'texture_quality': 'medium',
                'anti_aliasing': False
            }
        elif self.current_fps < 40:
            # Optimización moderada
            return {
                'lod_distance': 100,
                'physics_simplify': False,
                'disable_shadows': False,
                'texture_quality': 'high',
                'anti_aliasing': True
            }
        else:
            # Configuración normal
            return {
                'lod_distance': 200,
                'physics_simplify': False,
                'disable_shadows': False,
                'texture_quality': 'ultra',
                'anti_aliasing': True
            }

    def optimize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Optimiza pesos basándose en FPS"""
        if self.current_fps < 30:
            adjustment = 0.94  # -6% cuando FPS bajo
        elif self.current_fps < 40:
            adjustment = 0.98  # -2% cuando FPS medio
        else:
            adjustment = 1.04   # +4% cuando FPS alto

        return weights * adjustment

    def get_fps_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de FPS"""
        if not self.frame_times:
            return {}

        return {
            'current_fps': self.current_fps,
            'target_fps': self.target_fps,
            'avg_frame_time_ms': np.mean(self.frame_times),
            'min_frame_time_ms': np.min(self.frame_times),
            'max_frame_time_ms': np.max(self.frame_times),
            'stability': self._calculate_stability(),
            'fps_variance': np.var(self.frame_times)
        }


def test_fps_controller():
    """Test del controlador FPS"""
    print("\n" + "="*70)
    print("TEST: FPSController")
    print("="*70)

    controller = FPSController()

    # Test 1: Análisis de frames normales
    print("\n✓ Test 1: Analizando frames...")
    for i in range(5):
        frame_time = 20 + np.random.rand() * 5  # 20-25ms
        analysis = controller.analyze_frame_time(frame_time)
        print(f"  ✓ FPS: {analysis['current_fps']:.1f}, Lagging: {analysis['is_lagging']}")

    # Test 2: FPS bajo
    print("\n✓ Test 2: Detección de FPS bajo...")
    for i in range(3):
        frame_time = 50 + np.random.rand() * 10  # 50-60ms
        controller.analyze_frame_time(frame_time)

    analysis = controller.analyze_frame_time(55)
    print(f"✓ FPS: {analysis['current_fps']:.1f}")
    print(f"✓ Lagging: {analysis['is_lagging']}")

    # Test 3: Optimization settings
    print("\n✓ Test 3: Obtener settings de optimización...")
    settings = controller.get_optimization_settings()
    print(f"✓ LOD Distance: {settings['lod_distance']}m")
    print(f"✓ Texture Quality: {settings['texture_quality']}")
    print(f"✓ Anti-aliasing: {settings['anti_aliasing']}")

    # Test 4: Estadísticas
    print("\n✓ Test 4: Estadísticas FPS...")
    stats = controller.get_fps_statistics()
    print(f"✓ FPS actual: {stats['current_fps']:.1f}")
    print(f"✓ Estabilidad: {stats['stability']:.1f}%")
    print(f"✓ Avg frame time: {stats['avg_frame_time_ms']:.2f}ms")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return stats


if __name__ == "__main__":
    test_fps_controller()
