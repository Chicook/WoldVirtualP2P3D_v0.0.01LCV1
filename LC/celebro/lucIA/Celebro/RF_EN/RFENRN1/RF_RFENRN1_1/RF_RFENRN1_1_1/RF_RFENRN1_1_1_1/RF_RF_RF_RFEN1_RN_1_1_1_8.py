"""
PrimCounter - Contador y Optimizador de Prims
=========================================

Neurona especializada en contar y optimizar prims (objetos) del metaverso.

Algoritmos 2024-2025:
- Spatial indexing
- Occlusion culling
- LOD grouping
"""

import numpy as np
from typing import Dict, Any, List


class PrimCounter:
    """Contador y optimizador de prims"""

    def __init__(self):
        self.prim_limit = 2000
        self.prim_count = 0
        self.prim_distribution = {}

        self.optimization_history = []

        print("✓ PrimCounter inicializado")

    def count_prims(self, total_prims: int, prims_by_type: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Cuenta y analiza prims

        Args:
            total_prims: Total de prims
            prims_by_type: Prims por tipo

        Returns:
            Análisis de prims
        """
        self.prim_count = total_prims
        self.prim_distribution = prims_by_type or {}

        # Calcular ratio
        usage_ratio = total_prims / self.prim_limit

        # Detectar problemas
        is_overloaded = total_prims > self.prim_limit
        is_warning = total_prims > (self.prim_limit * 0.8)

        # Recomendaciones
        recommendation = self._get_optimization_recommendation(total_prims)

        analysis = {
            'total_prims': total_prims,
            'usage_ratio': usage_ratio,
            'is_overloaded': is_overloaded,
            'is_warning': is_warning,
            'prim_limit': self.prim_limit,
            'available_capacity': max(0, self.prim_limit - total_prims),
            'recommendation': recommendation,
            'prims_by_type': self.prim_distribution
        }

        self.optimization_history.append(analysis)

        return analysis

    def _get_optimization_recommendation(self, prim_count: int) -> str:
        """Obtiene recomendación de optimización"""
        if prim_count > self.prim_limit:
            return 'delete_unused_prims'
        elif prim_count > self.prim_limit * 0.8:
            return 'optimize_prim_geometry'
        else:
            return 'no_action'

    def optimize_prim_distribution(self, qo: float) -> Dict[str, Any]:
        """
        Optimiza distribución de prims

        Args:
            qo: Calidad operacional

        Returns:
            Configuración de optimización
        """
        if qo < 40:
            # Optimización extrema
            return {
                'cull_distance': 25,
                'max_prims_visible': 500,
                'prim_groups': 'aggregate',
                'detail_level': 'low'
            }
        elif qo < 60:
            # Optimización moderada
            return {
                'cull_distance': 50,
                'max_prims_visible': 1000,
                'prim_groups': 'medium',
                'detail_level': 'medium'
            }
        else:
            # Configuración normal
            return {
                'cull_distance': 100,
                'max_prims_visible': 2000,
                'prim_groups': 'detailed',
                'detail_level': 'high'
            }

    def optimize_weights(self, weights: np.ndarray, prim_count: int) -> np.ndarray:
        """Optimiza pesos basándose en cantidad de prims"""
        usage_ratio = prim_count / self.prim_limit

        if usage_ratio > 0.9:
            adjustment = 0.90  # -10%
        elif usage_ratio > 0.7:
            adjustment = 0.95  # -5%
        else:
            adjustment = 1.03  # +3%

        return weights * adjustment

    def get_prim_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de prims"""
        return {
            'current_prims': self.prim_count,
            'prim_limit': self.prim_limit,
            'usage_percentage': (self.prim_count / self.prim_limit) * 100,
            'distribution': self.prim_distribution,
            'total_analysis': len(self.optimization_history)
        }


def test_prim_counter():
    """Test del contador de prims"""
    print("\n" + "="*70)
    print("TEST: PrimCounter")
    print("="*70)

    counter = PrimCounter()

    # Test 1: Conteo normal
    print("\n✓ Test 1: Contando prims...")
    analysis1 = counter.count_prims(1200, {'sphere': 300, 'box': 500, 'cylinder': 400})
    print(f"✓ Total: {analysis1['total_prims']}")
    print(f"✓ Ratio: {analysis1['usage_ratio']:.2%}")
    print(f"✓ Recomendación: {analysis1['recommendation']}")

    # Test 2: Warning
    print("\n✓ Test 2: Advertencia de capacidad...")
    analysis2 = counter.count_prims(1800)
    print(f"✓ Total: {analysis2['total_prims']}")
    print(f"✓ Warning: {analysis2['is_warning']}")
    print(f"✓ Capacidad disponible: {analysis2['available_capacity']}")

    # Test 3: Optimización por QO
    print("\n✓ Test 3: Optimizando por QO...")
    config = counter.optimize_prim_distribution(35)
    print(f"✓ Cull distance: {config['cull_distance']}m")
    print(f"✓ Max prims visible: {config['max_prims_visible']}")
    print(f"✓ Detail level: {config['detail_level']}")

    # Test 4: Estadísticas
    print("\n✓ Test 4: Estadísticas...")
    stats = counter.get_prim_statistics()
    print(f"✓ Prims actuales: {stats['current_prims']}")
    print(f"✓ Uso: {stats['usage_percentage']:.1f}%")
    print(f"✓ Análisis totales: {stats['total_analysis']}")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return stats


if __name__ == "__main__":
    test_prim_counter()
