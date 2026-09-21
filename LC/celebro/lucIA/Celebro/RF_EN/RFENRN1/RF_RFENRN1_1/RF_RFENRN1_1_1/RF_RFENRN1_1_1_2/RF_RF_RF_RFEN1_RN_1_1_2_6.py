"""
P2PNetworkOptimizerNeuron - Optimizador de Red P2P
============================================

Neurona especializada en optimizar el ancho de banda de la red P2P según métricas.

Funcionalidad:
- Optimizar ancho de banda según QO
- Reducir tasa de seeding cuando latencia alta
- Gestión de recursos de red
"""

import numpy as np
from typing import Dict, Any, List


class P2PNetworkOptimizerNeuron:
    """Optimiza la red P2P según métricas"""

    def __init__(self):
        self.bandwidth_history = []
        self.optimization_applied = []

        print("✓ P2PNetworkOptimizerNeuron inicializado")

    def optimize_bandwidth(self, quality_operational: float,
                           current_bandwidth_mbps: float) -> Dict[str, Any]:
        """
        Optimiza ancho de banda según QO

        Args:
            quality_operational: Calidad operacional (0-100)
            current_bandwidth_mbps: Ancho de banda actual

        Returns:
            Configuración de optimización
        """
        if quality_operational < 40:
            # QO baja: reducir seeding agresivamente
            optimized_bandwidth = current_bandwidth_mbps * 0.3
            action = 'reduce_seeding_aggressive'
        elif quality_operational < 60:
            # QO media: reducir seeding moderadamente
            optimized_bandwidth = current_bandwidth_mbps * 0.6
            action = 'reduce_seeding_moderate'
        else:
            # QO alta: ancho de banda normal
            optimized_bandwidth = current_bandwidth_mbps * 1.0
            action = 'maintain_bandwidth'

        optimization = {
            'qo': quality_operational,
            'current_bandwidth_mbps': current_bandwidth_mbps,
            'optimized_bandwidth_mbps': optimized_bandwidth,
            'action': action,
            'reduction_percentage': ((current_bandwidth_mbps - optimized_bandwidth) / current_bandwidth_mbps) * 100
        }

        self.optimization_applied.append(optimization)
        return optimization

    def optimize_weights(self, weights: np.ndarray, qo: float) -> np.ndarray:
        """Optimiza pesos según QO"""
        if qo < 40:
            return weights * 0.90  # -10%
        elif qo < 60:
            return weights * 0.95  # -5%
        else:
            return weights * 1.05   # +5%

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        if not self.optimization_applied:
            return {}

        return {
            'total_optimizations': len(self.optimization_applied),
            'avg_qo': np.mean([opt['qo'] for opt in self.optimization_applied]),
            'avg_bandwidth_reduction': np.mean([opt['reduction_percentage']
                                               for opt in self.optimization_applied])
        }


def test_p2p_network_optimizer_neuron():
    """Test de la neurona optimizadora P2P"""
    print("\n" + "="*70)
    print("TEST: P2PNetworkOptimizerNeuron")
    print("="*70)

    neuron = P2PNetworkOptimizerNeuron()

    # Test 1: QO Baja
    print("\n✓ Test 1: Optimizando con QO baja...")
    opt1 = neuron.optimize_bandwidth(30.0, 10.0)
    print(f"✓ QO: {opt1['qo']:.1f}")
    print(f"✓ Bandwidth actual: {opt1['current_bandwidth_mbps']}Mbps")
    print(f"✓ Bandwidth optimizado: {opt1['optimized_bandwidth_mbps']:.1f}Mbps")
    print(f"✓ Acción: {opt1['action']}")

    # Test 2: QO Alta
    print("\n✓ Test 2: Optimizando con QO alta...")
    opt2 = neuron.optimize_bandwidth(80.0, 10.0)
    print(f"✓ Reducción: {opt2['reduction_percentage']:.1f}%")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return opt1


if __name__ == "__main__":
    test_p2p_network_optimizer_neuron()
