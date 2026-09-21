"""
BandwidthAllocatorNeuron - Asignador de Ancho de Banda
==============================================

Neurona especializada en asignar ancho de banda de forma inteligente.

Funcionalidad:
- Asignación dinámica de ancho de banda
- Priorización de traffic
- Gestión de cuotas
"""

import numpy as np
from typing import Dict, Any, List


class BandwidthAllocatorNeuron:
    """Asigna ancho de banda de forma inteligente"""

    def __init__(self):
        self.allocation_history = []
        self.total_bandwidth_mbps = 100.0

        print("✓ BandwidthAllocatorNeuron inicializado")

    def allocate_bandwidth(self, requests: List[Dict[str, Any]],
                           quality_operational: float) -> Dict[str, Any]:
        """
        Asigna ancho de banda según prioridad y QO

        Args:
            requests: Solicitudes de ancho de banda
            quality_operational: Calidad operacional

        Returns:
            Asignaciones de ancho de banda
        """
        allocations = []
        total_allocated = 0.0

        # Ordenar por prioridad
        sorted_requests = sorted(requests, key=lambda r: r.get('priority', 0), reverse=True)

        available_bandwidth = self.total_bandwidth_mbps

        # Ajustar según QO
        if quality_operational < 40:
            available_bandwidth *= 0.5
        elif quality_operational < 60:
            available_bandwidth *= 0.7
        elif quality_operational < 80:
            available_bandwidth *= 0.9

        for request in sorted_requests:
            requested = request.get('bandwidth_mbps', 0)
            priority = request.get('priority', 1)

            # Asignar según prioridad
            if priority == 'critical':
                allocated = min(requested, available_bandwidth * 0.4)
            elif priority == 'high':
                allocated = min(requested, available_bandwidth * 0.3)
            else:
                allocated = min(requested, available_bandwidth * 0.2)

            allocations.append({
                'request_id': request['id'],
                'requested_mbps': requested,
                'allocated_mbps': allocated,
                'priority': priority
            })

            total_allocated += allocated
            available_bandwidth -= allocated

        result = {
            'total_bandwidth_mbps': self.total_bandwidth_mbps,
            'allocated_bandwidth_mbps': total_allocated,
            'utilization_percent': (total_allocated / self.total_bandwidth_mbps) * 100,
            'allocations': allocations,
            'qo': quality_operational
        }

        self.allocation_history.append(result)
        return result

    def optimize_weights(self, weights: np.ndarray, utilization: float) -> np.ndarray:
        """Optimiza pesos según utilización"""
        if utilization > 80:
            return weights * 0.93  # -7%
        elif utilization > 60:
            return weights * 0.97  # -3%
        else:
            return weights * 1.05  # +5%

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        if not self.allocation_history:
            return {}

        utilizations = [h['utilization_percent'] for h in self.allocation_history]
        return {
            'total_allocations': len(self.allocation_history),
            'avg_utilization': np.mean(utilizations),
            'max_utilization': np.max(utilizations),
            'min_utilization': np.min(utilizations)
        }


def test_bandwidth_allocator_neuron():
    """Test de la neurona de asignación de ancho de banda"""
    print("\n" + "="*70)
    print("TEST: BandwidthAllocatorNeuron")
    print("="*70)

    neuron = BandwidthAllocatorNeuron()

    # Test 1: Asignar ancho de banda
    print("\n✓ Test 1: Asignando ancho de banda...")
    requests = [
        {'id': 'req_1', 'bandwidth_mbps': 20, 'priority': 'critical'},
        {'id': 'req_2', 'bandwidth_mbps': 15, 'priority': 'high'},
        {'id': 'req_3', 'bandwidth_mbps': 10, 'priority': 'normal'}
    ]

    result = neuron.allocate_bandwidth(requests, 45.0)
    print(f"✓ Bandwidth total: {result['total_bandwidth_mbps']}Mbps")
    print(f"✓ Asignado: {result['allocated_bandwidth_mbps']:.1f}Mbps")
    print(f"✓ Utilización: {result['utilization_percent']:.1f}%")

    for alloc in result['allocations']:
        print(f"  • {alloc['request_id']}: {alloc['allocated_mbps']:.1f}Mbps ({alloc['priority']})")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return result


if __name__ == "__main__":
    test_bandwidth_allocator_neuron()
