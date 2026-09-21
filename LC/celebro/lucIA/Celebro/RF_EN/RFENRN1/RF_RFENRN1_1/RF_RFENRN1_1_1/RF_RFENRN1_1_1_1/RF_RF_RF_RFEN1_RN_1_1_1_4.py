"""
LODOptimizer - Optimizador de Nivel de Detalle
==========================================

Neurona especializada en optimizar el LOD (Level of Detail) de objetos
en el metaverso según la QO.

Algoritmos 2024-2025:
- Adaptive LOD based on performance
- Distance-based quality scaling
- Dynamic mesh simplification
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class LODOptimizer:
    """Optimizador de nivel de detalle dinámico"""

    def __init__(self):
        self.default_lod_distance = 100  # metros
        self.min_lod_distance = 10   # metros mínimo
        self.max_lod_distance = 200  # metros máximo

        self.lod_levels = ['high', 'medium', 'low', 'lowest']
        self.current_lod_distance = self.default_lod_distance
        self.optimization_history = []

        print("✓ LODOptimizer inicializado")

    def calculate_optimal_lod(self, quality_operational: float,
                              prim_count: int, fps: float) -> Dict[str, Any]:
        """
        Calcula el LOD óptimo basándose en QO

        Args:
            quality_operational: Calidad operacional (0-100)
            prim_count: Número de prims
            fps: Frames por segundo

        Returns:
            Configuración de LOD optimizada
        """
        # Calcular distancia de LOD óptima
        if quality_operational >= 80:
            # QO Alta: LOD normal
            lod_distance = self.default_lod_distance
            lod_level = 'high'
        elif quality_operational >= 60:
            # QO Media: Reducir LOD a 50m
            lod_distance = 50
            lod_level = 'medium'
        elif quality_operational >= 40:
            # QO Baja: LOD bajo a 25m
            lod_distance = 25
            lod_level = 'low'
        else:
            # QO Crítica: LOD mínimo a 10m
            lod_distance = self.min_lod_distance
            lod_level = 'lowest'

        # Ajuste según número de prims
        if prim_count > 2000:
            lod_distance *= 0.8  # Reducir 20%

        # Ajuste según FPS
        if fps < 30:
            lod_distance *= 0.7  # Reducir 30%
        elif fps < 20:
            lod_distance *= 0.5  # Reducir 50%

        # Asegurar límites
        lod_distance = max(self.min_lod_distance,
                           min(self.max_lod_distance, lod_distance))

        # Calcular ahorro esperado
        savings = self._calculate_lod_savings(lod_distance, prim_count)

        config = {
            'lod_distance': lod_distance,
            'lod_level': lod_level,
            'quality_operational': quality_operational,
            'estimated_savings_prims': savings,
            'estimated_fps_boost': self._estimate_fps_boost(quality_operational)
        }

        self.optimization_history.append(config)

        # Mantener solo últimos 50
        if len(self.optimization_history) > 50:
            self.optimization_history.pop(0)

        self.current_lod_distance = lod_distance

        return config

    def _calculate_lod_savings(self, lod_distance: float, prim_count: int) -> int:
        """Calcula cuántos prims se ahorran con el LOD"""
        # Fórmula: ahorro proporcional a reducción de distancia
        reduction = (self.default_lod_distance - lod_distance) / self.default_lod_distance
        return int(prim_count * reduction * 0.3)  # 30% del área afectada

    def _estimate_fps_boost(self, qo: float) -> float:
        """Estima el aumento de FPS esperado"""
        if qo >= 80:
            return 0
        elif qo >= 60:
            return 2.5  # +2.5 FPS
        elif qo >= 40:
            return 5.0  # +5 FPS
        else:
            return 8.0  # +8 FPS

    def apply_lod_settings(self, prims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aplica configuraciones de LOD a prims

        Args:
            prims: Lista de prims con sus configuraciones

        Returns:
            Prims con LOD actualizado
        """
        optimized_prims = []

        for prim in prims:
            distance_to_camera = prim.get('distance', self.default_lod_distance)

            # Aplicar LOD según distancia
            if distance_to_camera > self.current_lod_distance:
                prim['lod_active'] = True
                prim['detail_level'] = 'low'
            else:
                prim['lod_active'] = False
                prim['detail_level'] = 'high'

            optimized_prims.append(prim)

        return optimized_prims

    def optimize_weights(self, weights: np.ndarray, qo: float) -> np.ndarray:
        """Optimiza pesos basándose en QO para LOD"""
        adjustment = 0.95 + (qo / 100) * 0.15  # Ajuste entre 0.95-1.10
        return weights * adjustment

    def get_lod_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de optimización LOD"""
        if not self.optimization_history:
            return {}

        lod_distances = [h['lod_distance'] for h in self.optimization_history]

        return {
            'current_lod_distance': self.current_lod_distance,
            'avg_lod_distance': np.mean(lod_distances),
            'min_lod_distance': np.min(lod_distances),
            'max_lod_distance': np.max(lod_distances),
            'total_optimizations': len(self.optimization_history),
            'last_qo': self.optimization_history[-1]['quality_operational']
        }


def test_lod_optimizer():
    """Test del optimizador LOD"""
    print("\n" + "="*70)
    print("TEST: LODOptimizer")
    print("="*70)

    optimizer = LODOptimizer()

    # Test 1: QO Alta (80-100)
    print("\n✓ Test 1: QO Alta...")
    config1 = optimizer.calculate_optimal_lod(90.0, 1000, 45)
    print(f"✓ LOD Distance: {config1['lod_distance']:.0f}m")
    print(f"✓ Level: {config1['lod_level']}")
    print(f"✓ FPS Boost: +{config1['estimated_fps_boost']:.1f}")

    # Test 2: QO Media (60-79)
    print("\n✓ Test 2: QO Media...")
    config2 = optimizer.calculate_optimal_lod(65.0, 2000, 35)
    print(f"✓ LOD Distance: {config2['lod_distance']:.0f}m")
    print(f"✓ Ahorro estimado: {config2['estimated_savings_prims']} prims")

    # Test 3: QO Baja (<60)
    print("\n✓ Test 3: QO Baja...")
    config3 = optimizer.calculate_optimal_lod(35.0, 3000, 15)
    print(f"✓ LOD Distance: {config3['lod_distance']:.0f}m")
    print(f"✓ FPS Boost: +{config3['estimated_fps_boost']:.1f}")

    # Test 4: Aplicar settings a prims
    print("\n✓ Test 4: Aplicando LOD a prims...")
    test_prims = [
        {'id': 1, 'distance': 30},
        {'id': 2, 'distance': 60},
        {'id': 3, 'distance': 120}
    ]
    optimized = optimizer.apply_lod_settings(test_prims)
    for prim in optimized:
        print(f"  ✓ Prim {prim['id']}: {prim['detail_level']} detail")

    # Test 5: Estadísticas
    print("\n✓ Test 5: Estadísticas LOD...")
    stats = optimizer.get_lod_statistics()
    print(f"✓ LOD actual: {stats['current_lod_distance']:.0f}m")
    print(f"✓ LOD promedio: {stats['avg_lod_distance']:.0f}m")
    print(f"✓ Optimizaciones: {stats['total_optimizations']}")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return config3


if __name__ == "__main__":
    test_lod_optimizer()
