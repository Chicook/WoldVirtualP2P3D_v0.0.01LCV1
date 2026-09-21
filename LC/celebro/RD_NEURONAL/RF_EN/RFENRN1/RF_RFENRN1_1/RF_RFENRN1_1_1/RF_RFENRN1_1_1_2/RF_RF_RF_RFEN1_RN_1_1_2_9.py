"""
NetworkLatencyPredictorNeuron - Predictor de Latencia de Red
=====================================================

Neurona especializada en predecir latencia de red para optimizar conexiones P2P.

Funcionalidad:
- Predicción de latencia
- Selección de mejores rutas
- Optimización proactiva
"""

import numpy as np
from typing import Dict, Any, List


class NetworkLatencyPredictorNeuron:
    """Predice latencia de red para optimización"""

    def __init__(self):
        self.prediction_history = []
        self.accuracy_stats = {'total': 0, 'correct': 0}

        print("✓ NetworkLatencyPredictorNeuron inicializado")

    def predict_latency(self, peer_distance_km: float,
                        network_load: float, historical_latency: List[float]) -> Dict[str, Any]:
        """
        Predice latencia hacia un peer

        Args:
            peer_distance_km: Distancia al peer
            network_load: Carga de red (0-1)
            historical_latency: Historial de latencia

        Returns:
            Predicción de latencia
        """
        # Predicción basada en distancia y carga
        base_latency = 10 + (peer_distance_km * 0.5)
        load_factor = 1.0 + network_load * 1.5

        predicted_latency_ms = base_latency * load_factor

        # Usar historial si existe
        if historical_latency:
            historical_avg = np.mean(historical_latency)
            predicted_latency_ms = predicted_latency_ms * 0.7 + historical_avg * 0.3

        prediction = {
            'peer_distance_km': peer_distance_km,
            'network_load': network_load,
            'predicted_latency_ms': predicted_latency_ms,
            'confidence': 0.85 if historical_latency else 0.65
        }

        self.prediction_history.append(prediction)
        self.accuracy_stats['total'] += 1

        return prediction

    def optimize_route_selection(self, peer_options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Selecciona la mejor ruta basándose en predicciones

        Args:
            peer_options: Opciones de peers

        Returns:
            Mejor peer seleccionado
        """
        predictions = []
        for peer in peer_options:
            pred = self.predict_latency(
                peer.get('distance_km', 1000),
                peer.get('network_load', 0.5),
                peer.get('historical_latency', [])
            )
            peer['predicted_latency'] = pred['predicted_latency_ms']
            predictions.append(pred['predicted_latency_ms'])

        # Seleccionar peer con menor latencia predicha
        best_idx = np.argmin(predictions)
        best_peer = peer_options[best_idx]

        return {
            'selected_peer': best_peer['id'],
            'predicted_latency_ms': predictions[best_idx],
            'alternative_peers': len(peer_options) - 1
        }

    def optimize_weights(self, weights: np.ndarray, accuracy: float) -> np.ndarray:
        """Optimiza pesos según precisión"""
        if accuracy > 0.8:
            return weights * 1.08  # +8%
        elif accuracy > 0.6:
            return weights * 1.03  # +3%
        else:
            return weights * 0.95  # -5%

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        return {
            'total_predictions': self.accuracy_stats['total'],
            'accuracy_rate': self.accuracy_stats['correct'] / self.accuracy_stats['total']
            if self.accuracy_stats['total'] > 0 else 0
        }


def test_network_latency_predictor_neuron():
    """Test de la neurona de predicción de latencia"""
    print("\n" + "="*70)
    print("TEST: NetworkLatencyPredictorNeuron")
    print("="*70)

    neuron = NetworkLatencyPredictorNeuron()

    # Test 1: Predecir latencia
    print("\n✓ Test 1: Prediciendo latencia...")
    pred = neuron.predict_latency(100.0, 0.3, [25, 27, 30, 26])
    print(f"✓ Latencia predicha: {pred['predicted_latency_ms']:.1f}ms")
    print(f"✓ Confianza: {pred['confidence']*100:.1f}%")

    # Test 2: Seleccionar ruta
    print("\n✓ Test 2: Seleccionando mejor ruta...")
    peers = [
        {'id': 'peer_1', 'distance_km': 50, 'network_load': 0.2, 'historical_latency': [20, 22]},
        {'id': 'peer_2', 'distance_km': 100, 'network_load': 0.5, 'historical_latency': [30, 35]}
    ]

    best = neuron.optimize_route_selection(peers)
    print(f"✓ Peer seleccionado: {best['selected_peer']}")
    print(f"✓ Latencia predicha: {best['predicted_latency_ms']:.1f}ms")

    print("\n✅ Tests completados exitosamente")
    print("="*70 + "\n")

    return best


if __name__ == "__main__":
    test_network_latency_predictor_neuron()
