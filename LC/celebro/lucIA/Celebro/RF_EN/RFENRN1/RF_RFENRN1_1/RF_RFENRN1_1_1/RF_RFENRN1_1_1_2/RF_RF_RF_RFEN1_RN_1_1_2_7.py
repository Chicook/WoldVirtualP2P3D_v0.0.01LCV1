"""
PeerPrioritizationNeuron - Priorización de Peers
==========================================

Neurona especializada en priorizar peers según latencia y confianza.

Funcionalidad:
- Asignar puntuación de confianza
- Priorizar peers con baja latencia
- Ordenar peers por rendimiento
"""

import numpy as np
from typing import Dict, Any, List


class PeerPrioritizationNeuron:
    """Prioriza peers para descarga de assets"""

    def __init__(self):
        self.peer_scores = {}
        self.prioritization_history = []

        print("✓ PeerPrioritizationNeuron inicializado")

    def prioritize_peers(self, peers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioriza peers según latencia y confianza

        Args:
            peers: Lista de peers con información

        Returns:
            Peers ordenados por prioridad
        """
        for peer in peers:
            peer_id = peer['id']

            # Calcular score
            latency_score = 100 / (peer.get('latency_ms', 1) + 1)
            trust_score = peer.get('trust_score', 0.5) * 50
            uptime_score = peer.get('uptime_percent', 50) / 100 * 30

            total_score = latency_score + trust_score + uptime_score

            peer['priority_score'] = total_score
            self.peer_scores[peer_id] = total_score

        # Ordenar por score
        sorted_peers = sorted(peers, key=lambda p: p['priority_score'], reverse=True)

        self.prioritization_history.append({
            'peers_count': len(peers),
            'top_peer': sorted_peers[0]['id'] if sorted_peers else None
        })

        return sorted_peers

    def optimize_weights(self, weights: np.ndarray, best_peer_latency: float) -> np.ndarray:
        """Optimiza pesos según latencia del mejor peer"""
        if best_peer_latency < 50:
            return weights * 1.08  # +8%
        elif best_peer_latency < 100:
            return weights * 1.03  # +3%
        else:
            return weights * 0.97  # -3%

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        return {
            'total_peers_scored': len(self.peer_scores),
            'total_prioritizations': len(self.prioritization_history),
            'best_score': max(self.peer_scores.values()) if self.peer_scores else 0,
            'worst_score': min(self.peer_scores.values()) if self.peer_scores else 0
        }


def test_peer_prioritization_neuron():
    """Test de la neurona de priorización de peers"""
    print("\n" + "="*70)
    print("TEST: PeerPrioritizationNeuron")
    print("="*70)

    neuron = PeerPrioritizationNeuron()

    # Test 1: Priorizar peers
    print("\n✓ Test 1: Priorizando peers...")
    peers = [
        {'id': 'peer_1', 'latency_ms': 50, 'trust_score': 0.9, 'uptime_percent': 95},
        {'id': 'peer_2', 'latency_ms': 100, 'trust_score': 0.7, 'uptime_percent': 80},
        {'id': 'peer_3', 'latency_ms': 200, 'trust_score': 0.5, 'uptime_percent': 60}
    ]

    prioritized = neuron.prioritize_peers(peers)

    for i, peer in enumerate(prioritized, 1):
        print(f"  {i}. {peer['id']}: Score={peer['priority_score']:.1f}")

    # Test 2: Estadísticas
    print("\n✓ Test 2: Estadísticas...")
    stats = neuron.get_statistics()
    print(f"✓ Peers scored: {stats['total_peers_scored']}")
    print(f"✓ Mejor score: {stats['best_score']:.1f}")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return prioritized


if __name__ == "__main__":
    test_peer_prioritization_neuron()
